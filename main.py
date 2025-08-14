import os
import subprocess
import requests
from moviepy.editor import VideoFileClip, concatenate_videoclips
from scenedetect import VideoManager, SceneManager
from scenedetect.detectors import ContentDetector

# Install dependencies if missing
try:
    import moviepy
    import scenedetect
except ImportError:
    subprocess.check_call(["pip", "install", "requests", "moviepy", "scenedetect"])

# Hugging Face API Key
HF_API_KEY = "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"
MODEL_NAME = "openai/clip-vit-base-patch32"

def detect_scenes(video_path):
    video_manager = VideoManager([video_path])
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=30.0))
    video_manager.start()
    scene_manager.detect_scenes(frame_source=video_manager)
    return scene_manager.get_scene_list()

def classify_scene(frame_path):
    api_url = f"https://api-inference.huggingface.co/models/{MODEL_NAME}"
    headers = {"Authorization": f"Bearer {HF_API_KEY}"}
    with open(frame_path, "rb") as f:
        img_bytes = f.read()
    resp = requests.post(api_url, headers=headers, data=img_bytes)
    try:
        return resp.json()
    except:
        return None

def create_trailer(input_video, output_trailer):
    scenes = detect_scenes(input_video)
    selected_clips = []
    for i, (start, end) in enumerate(scenes):
        clip = VideoFileClip(input_video).subclip(start.get_seconds(), end.get_seconds())
        frame_path = f"frame_{i}.jpg"
        clip.save_frame(frame_path, t=0.5)
        labels = classify_scene(frame_path)
        if labels:
            text = str(labels).lower()
            if "fight" in text or "emotional" in text or "moral" in text:
                selected_clips.append(clip)
    if selected_clips:
        final_clip = concatenate_videoclips(selected_clips)
        final_clip = final_clip.subclip(0, min(final_clip.duration, 150))
        final_clip.write_videofile(output_trailer)
    else:
        print("No matching scenes found!")

if __name__ == "__main__":
    if not os.path.exists("movie.mp4"):
        print("❌ Please put your movie file as 'movie.mp4' in the same folder.")
    else:
        create_trailer("movie.mp4", "trailer.mp4")
        print("✅ Trailer saved as trailer.mp4")
