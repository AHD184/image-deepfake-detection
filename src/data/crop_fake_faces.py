from pathlib import Path
import cv2
import mediapipe as mp

input_dir = Path("data/interim/fake_frames")

output_dir = Path("data/interim/fake_faces")
output_dir.mkdir(parents=True, exist_ok=True)

# MediaPipe setup
BaseOptions = mp.tasks.BaseOptions
FaceDetector = mp.tasks.vision.FaceDetector
FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
VisionRunningMode = mp.tasks.vision.RunningMode

detector_options = FaceDetectorOptions(
    base_options=BaseOptions(model_asset_path="checkpoints/face_detector.task"), # Specify which trained model we're using
    running_mode=VisionRunningMode.IMAGE, # We're processing images, not video/live stream
    min_detection_confidence=0.5
)

detector = FaceDetector.create_from_options(detector_options)

# image_paths is a list of Path objects from which we'll read each image in the next couple lines
image_paths = sorted(input_dir.glob("*/*.jpg")) # Opens each subfolder and finds all .jpg files and order them

for image_path in image_paths:
    image = cv2.imread(str(image_path)) # OpenCV loads image in BGR format but MP needs it in RGB

    if image is None:
        print(f"Couldn't read: {image_path}")
        continue

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # We convert NumPy array (image_rbg) to MediaPipe's own image format (Standard RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    results = detector.detect(mp_image)

    if not results.detections: # results is on object returned by the detector and detections is the obj'd attribute
        print(f"No face found: {image_path.name}")
        continue

    detection = results.detections[0]
    bbox = detection.bounding_box

    h, w, _ = image.shape # We don't care about number of channel (RGB has 3)

    # To prevent errors
    x = max(0, bbox.origin_x)
    y = max(0, bbox.origin_y)
    box_w = bbox.width
    box_h = bbox.height

    box_w = min(box_w, w - x)
    box_h = min(box_h, h - y)

    face_crop = image[y:y + box_h, x:x + box_w]

    if face_crop.size == 0:
        print(f"Empty crop: {image_path.name}")
        continue

    video_folder = image_path.parent.name
    save_folder = output_dir / video_folder
    save_folder.mkdir(parents=True, exist_ok=True)

    save_path = save_folder / image_path.name

    cv2.imwrite(str(save_path), face_crop)
    print(f"Saved: {save_path}")

detector.close()
print("Done cropping fake faces.")