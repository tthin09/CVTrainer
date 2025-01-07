import cv2 as cv
import mediapipe as mp
from image_processor import ImageProcessor
import argparse

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

image_processor = ImageProcessor()

# Parse arguments
parser = argparse.ArgumentParser(description="Process camera or video input")
parser.add_argument("--source",
                    type=str,
                    default="camera",
                    choices=["camera", "video"],
                    help="Source of input: 'camera' or 'file'")
parser.add_argument("--video_path",
                    type=str,
                    default="videos/curls_1.mp4",
                    help="Path to the video file")

cap = cv.VideoCapture("videos/curls_2.mp4")
is_video = False
args = parser.parse_args()
if args.source == "camera":
  cap = cv.VideoCapture(0)
elif args.source == "video":
  is_video = True
  if not args.video_path:
    print("Require a video_path input")
    exit(1)
  cap = cv.VideoCapture(args.video_path)


with mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as pose:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      if is_video:
        image_processor.end()
        break
      continue
    
    # preprocess frame
    height, width, temp = image.shape
    image = cv.resize(image, (1280, 720))
    temp = 200
    image = image[:, temp:(1280-temp)]
    image = cv.flip(image, 1)

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    results = pose.process(image)
    image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
    if not results.pose_landmarks:
      continue
        
    image_processor.loadImage(image, results.pose_landmarks)
    image_processor.process() 
      
    cv.imshow('MediaPipe Pose', image)
    key = cv.waitKey(5)
    if key & 0xFF == 27: # ESC key
      image_processor.end()
      break
    if key & 0xFF == 32: # Space key
      image_processor.toggleMode()
    
  
cap.release()

