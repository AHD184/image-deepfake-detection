import cv2
from pathlib import Path # We do this to avoid writing manually the entire address (messy strings)

def extract_frames_from_video(video_path, output_folder, frame_skip=10):
    # Extract every 10th frame from a video and save it as a jpg

    cap = cv2.VideoCapture(str(video_path)) # Input will be Path(address), so we do str() to convert it back

    if not cap.isOpened():
        print(f"Could not open video: {video_path}")
        return
    
    # This creates all the folders that are missing in the path chain, doesn't crash if they already exist
    # We make a folder for each video, this creates a path for these folders
    output_folder.mkdir(parents=True, exist_ok=True)

    frame_index = 0
    saved_count = 0

    # ret is a boolean which returns false if a frame wasn't captured
    while True:
        ret, frame = cap.read()

        if not ret:
            break
        
        if frame_index % frame_skip == 0:
            frame_name = f"frame_{frame_index:04d}.jpg" # 04d means we get four digits padded with 0s
            frame_path = output_folder / frame_name # Names a path for each image (adds it to the end)
            cv2.imwrite(str(frame_path), frame) # Writes image to disk, frame is the actual data which is a NumPy array
            saved_count += 1
        
        frame_index += 1
    
    cap.release()
    print(f"Saved {saved_count} frames from {video_path.name}") # video_path.name is the final part of the path

def main():
    input_dir = Path("data/raw/ffpp/original_sequences/youtube/c23/videos")
    output_dir = Path("data/interim/real_frames")

    # We make a list of all the files inside input_dir which have an .mp4 extension
    # We use glob() for this
    video_files = list(input_dir.glob("*.mp4")) 

    print(f"Found {len(video_files)} videos.") # We know there are 50 videos

    for video_path in video_files:
        # .stem removes extensions, so that we can create a path with only the video name in the next step
        video_id = video_path.stem 
        video_output_folder = output_dir / video_id

        extract_frames_from_video(video_path, video_output_folder, frame_skip=10)

    # Every .py file has a built_in variable called __name__
    # if we directly run this file, then __name__ = __main__, which is True
    # So python will run main(), which extracts frames from the dataset
    # Assume we want to reuse def extract_frames_from_video in another file
    # Then we do from extract_frames import extract_frames_from_video
    # In this case, __name__ = "extract_frames", therefore main won't run
    # And the dataset extraction won't happen, making only the function available

if __name__ == "__main__":
    main()