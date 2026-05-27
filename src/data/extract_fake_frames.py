from pathlib import Path
import cv2

# Path to the fake videos folder
fake_videos_dir = Path("data/raw/ffpp/manipulated_sequences/Deepfakes/c23/videos")

# Path to where the extracted frames will be saved
output_dir = Path("data/interim/fake_frames")

# Create output folder if it does not exist
# If parent folders are missing, it creates them too
output_dir.mkdir(parents=True, exist_ok=True) 

# Get all the .mp4 files
video_paths = sorted(fake_videos_dir.glob("*.mp4"))

frame_interval = 30

for video_path in video_paths:
    # Folder name based on video filename without extension
    video_name = video_path.stem 
    video_output_dir = output_dir / video_name 
    video_output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))

    frame_idx = 0
    saved_idx = 0

    while True:
        success, frame = cap.read() # frame is a NumPy array

        if not success:
            break

        if frame_idx % frame_interval == 0:
            frame_filename = f"frame_{saved_idx:04d}.jpg"
            frame_path = video_output_dir / frame_filename
            cv2.imwrite(str(frame_path), frame)
            saved_idx += 1
        
        frame_idx += 1

    cap.release()
    print(f"Done: {video_name} -> saved {saved_idx} frames")

print("Fake frame extraction complete.")