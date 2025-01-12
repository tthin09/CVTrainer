## Installation
You need to install OpenCV and MediaPipe
```cmd
pip install opencv-python
pip install mediapipe
```

## Usage
Run the main.py file in command line. We have 2 option: Using live camera or using videos downloaded in the "\\videos" folder.

Arguments:

- **--source:** 'camera' or 'video'.
- **--video_name:** name of the  video in "\\videos" folder. Leave this as blank if you're using live camera.

Example:
```cmd
python main.py --source=camera --video_name=curls_1.mp4
```
