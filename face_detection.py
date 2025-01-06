import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import time


cap = cv.VideoCapture(0)
face_cascade = cv.CascadeClassifier(cv.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv.CascadeClassifier(cv.data.haarcascades + "haarcascade_eye.xml")

SCALE_FACTOR = 1.15

count_image = 1

while True:
  ret, frame = cap.read()
  
  gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
  faces = face_cascade.detectMultiScale(gray, SCALE_FACTOR, 5)
  
  for (x, y, w, h) in faces:
    new_image = roi_color
    new_image_filename = "images/face_" + str(count_image) + ".jpg"
    cv.imwrite(new_image_filename, new_image)
    count_image += 1
  
  for (x, y, w, h) in faces:
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = frame[y:y+h, x:x+w]
    
    cv.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
    eyes = eye_cascade.detectMultiScale(roi_gray, 1.3, 5)
    for (ex, ey, ew, eh) in eyes:
      cv.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (255, 0, 0), 2)
  
  
  cv.imshow("frame", frame)
  if cv.waitKey(1) == ord('q'):
    break
  time.sleep(1)

  
cap.release()
cv.destroyAllWindows()