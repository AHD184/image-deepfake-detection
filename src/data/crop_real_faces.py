from pathlib import Path
import cv2
import mediapipe as mp  # Google library for face detection

input_dir = Path("data/interim/real_frames")

output_dir = Path("data/interim/real_faces")
output_dir.mkdir(parents=True, exist_ok=True)

# MediaPipe face detector setup
BaseOptions = mp.tasks.BaseOptions # Used to define basic settings and where the model file is
FaceDetector = mp.tasks.vision.FaceDetector # The detector class which will detect faces
FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions   # Used to configure the detector
VisionRunningMode = mp.tasks.vision.RunningMode # Tell MediaPipe how we're using it (only for images in our case)

detector_options = FaceDetectorOptions(
    base_options=BaseOptions(model_asset_path="checkpoints/face_detector.task"), # Specify the pretrained model to detect faces
    running_mode=VisionRunningMode.IMAGE,   # Model optimized for faces farther from camera
    min_detection_confidence=0.5    # Detect faces only if model >= 50% sure
)

detector = FaceDetector.create_from_options(detector_options)   # Builds the detector using the settings we defined

image_paths = sorted(input_dir.glob("*/*.jpg")) # Goes into eac subfolder and finds all .jpg files

for image_path in image_paths:
    image = cv2.imread(str(image_path)) #convert path to string and read image from disk and save to a NumPy array

    if image is None:
        print(f"Couldn't read: {image_path}")
        continue

    # OpenCV loads images in BGR but MediaPipe expects RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR_BGR2RGB is a flag that tells how to convert the image
    
    # Create an mp object from the NumPy array, image_format tells mp that image is in standard RGB format                            
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    results = detector.detect(mp_image)

    if not results.detections: 
        print(f"No face found: {image_path.name}")
        continue

    detection = results.detections[0] # If there are multiple faces, then take only the first face; can improvr later
    # Get a rectangle enclosing the face in pixels
    bbox = detection.bounding_box 

    # Channels is the number of color channels
    h, w, _ = image.shape # shape returns height, width and channels (we don't care about this value hence '_')

    x = max(0, bbox.origin_x)
    y = max(0, bbox.origin_y)
    box_w = bbox.width
    box_h = bbox.height

    # Make sure coordinates stay inside the image
    box_w = min(box_w, w - x)
    box_h = min(box_h, h - y)

    face_crop = image[y:y + box_h, x:x + box_w]

    if face_crop.size == 0:
        print(f"Empty crop: {image_path.name}")
        continue

    # Ex: real_frames/033/frame_0000.jpg, then video_folder = "033"
    video_folder = image_path.parent.name
    save_folder = output_dir / video_folder
    save_folder.mkdir(parents=True, exist_ok=True)
    # Cont. ex: save_path will be real_faces/033/frame_0000.jpg (including the extension)
    save_path = save_folder / image_path.name

    cv2.imwrite(str(save_path), face_crop)
    print(f"Saved: {save_path}")

detector.close()
print("Done cropping real faces.")
