
import os
import json
import math
import time
import tempfile
import requests
from pathlib import Path

from moviepy.editor import VideoFileClip, concatenate_videoclips, vfx
from scenedetect import VideoManager, SceneManager
from scenedetect.detectors import ContentDetector

# -------- Settings --------
MAX_TRAILER_SECONDS = 150  # final trailer cap
SCENE_DETECT_THRESHOLD = 27.0  # lower = more cuts detected
MIN_SCENE_SECONDS = 2.0  # ignore ultra-short scenes
MODEL_NAME = "microsoft/resnet-50"  # image-classification model (stable on HF)
HF_API_KEY = os.getenv("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", "").strip()  # set via GitHub Secret or env
INTEREST_KEYWORDS = [
    "fight", "violence", "gun", "weapon", "sword", "explosion",
    "car", "racing", "love", "kiss", "cry", "sad", "angry",
    "laugh", "crowd", "running", "fire", "police", "blood"
]
# --------------------------

def log(msg):
    print(f"[TrailerMaker] {msg}", flush=True)

def detect_scenes(video_path: str):
    """
    Detect scenes using PySceneDetect (ContentDetector).
    Returns list of (start_sec, end_sec).
    """
    log("Detecting scenes...")
    vm = VideoManager([video_path])
    sm = SceneManager()
    sm.add_detector(ContentDetector(threshold=SCENE_DETECT_THRESHOLD))
    vm.start()
    sm.detect_scenes(frame_source=vm)
    scene_list = sm.get_scene_list()
    vm.release()

    scenes = []
    for start, end in scene_list:
        s = start.get_seconds()
        e = end.get_seconds()
        if e - s >= MIN_SCENE_SECONDS:
            scenes.append((s, e))

    if not scenes:
        # Fallback: fixed segments if detection finds nothing
        log("No scenes detected; using fallback fixed segments of ~8s.")
        with VideoFileClip(video_path) as clip:
            dur = clip.duration
        step = 8.0
        t = 0.0
        while t < dur:
            s = t
            e = min(t + step, dur)
            if e - s >= MIN_SCENE_SECONDS:
                scenes.append((s, e))
            t += step

    log(f"Found {len(scenes)} scenes.")
    return scenes

def hf_classify_image(image_path: str):
    """
    Call Hugging Face Inference API (image-classification).
    Returns list of dicts [{label, score}, ...] or [] on failure.
    """
    if not HF_API_KEY:
        return []

    url = f"https://api-inference.huggingface.co/models/{MODEL_NAME}"
    headers = {"Authorization": f"Bearer {HF_API_KEY}"}
    try:
        with open(image_path, "rb") as f:
            data = f.read()
        r = requests.post(url, headers=headers, data=data, timeout=60)
        r.raise_for_status()
        out = r.json()
        if isinstance(out, dict) and out.get("error"):
            log(f"HF warning: {out.get('error')}")
            return []
        if isinstance(out, list):
            # Already in expected format
            return out
        return []
    except Exception as e:
        log(f"HF classify failed: {e}")
        return []

def score_scene_by_labels(labels_json) -> float:
    """
    Assign a score based on keyword presence & top probabilities.
    """
    if not labels_json:
        return 0.0
    text = " ".join([str(x.get("label", "")).lower() for x in labels_json])
    score = 0.0
    for kw in INTEREST_KEYWORDS:
        if kw in text:
            score += 1.0
    # Add a bit of weight for confidence of top-3
    top3 = sorted(labels_json, key=lambda x: x.get("score", 0), reverse=True)[:3]
    score += sum([x.get("score", 0) for x in top3]) * 0.5
    return score

def pick_interesting_scenes(video_path: str, scenes):
    """
    For each scene, snapshot a middle frame, classify via HF (if API key set),
    and compute a score. Fallback to duration-based score if HF missing.
    """
    log("Scoring scenes...")
    scores = []
    tmpdir = tempfile.mkdtemp(prefix="frames_")
    with VideoFileClip(video_path) as clip:
        for idx, (s, e) in enumerate(scenes):
            mid = s + (e - s) / 2
            frame_path = str(Path(tmpdir) / f"scene_{idx}.jpg")
            try:
                # Save a frame
                clip.save_frame(frame_path, t=max(s + 0.1, min(mid, clip.duration - 0.1)))
                labels = hf_classify_image(frame_path) if HF_API_KEY else []
                if labels:
                    sc = score_scene_by_labels(labels)
                else:
                    # Fallback: favor medium-length scenes
                    length = e - s
                    sc = min(length, 10.0) / 10.0  # 0..1
                scores.append((sc, s, e))
            except Exception as ex:
                log(f"Frame save/classify failed for scene {idx}: {ex}")
                # small fallback score if snapshot fails
                scores.append((0.1, s, e))
    scores.sort(reverse=True, key=lambda x: x[0])
    log("Scoring complete.")
    return scores

def build_trailer(video_path: str, ranked_scenes):
    """
    Concatenate best scenes up to MAX_TRAILER_SECONDS.
    """
    log("Building trailer...")
    selected_clips = []
    total = 0.0

    with VideoFileClip(video_path) as clip:
        for sc, s, e in ranked_scenes:
            dur = e - s
            if dur < MIN_SCENE_SECONDS:
                continue
            take = min(dur, max(6.0, min(12.0, dur)))  # keep reasonable chunk
            if total + take > MAX_TRAILER_SECONDS:
                take = MAX_TRAILER_SECONDS - total
            if take <= 0:
                break
            sub = clip.subclip(s, s + take)
            # Add gentle fade in/out
            sub = sub.fx(vfx.fadein, 0.3).fx(vfx.fadeout, 0.3)
            selected_clips.append(sub)
            total += take
            if total >= MAX_TRAILER_SECONDS:
                break

        if not selected_clips:
            log("No selected clips; copying first 20s as fallback.")
            sub = clip.subclip(0, min(20.0, clip.duration))
            selected_clips = [sub]

        final = concatenate_videoclips(selected_clips, method="compose")
        out = "trailer.mp4"
        # Write with safe defaults; relies on system ffmpeg or imageio-ffmpeg
        final.write_videofile(
            out,
            codec="libx264",
            audio_codec="aac",
            temp_audiofile="temp-audio.m4a",
            remove_temp=True,
            threads=2,
            fps=min(30, getattr(clip, "fps", 24) or 24),
        )
        log(f"Trailer ready: {out}")

def main():
    input_video = "movie.mp4"
    if not os.path.exists(input_video):
        log("❌ 'movie.mp4' not found in project root. Add your video and run again.")
        return

    scenes = detect_scenes(input_video)
    ranked = pick_interesting_scenes(input_video, scenes)
    build_trailer(input_video, ranked)
    log("✅ Done.")

if __name__ == "__main__":
    main()
