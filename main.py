import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import sys

filename = "minesweeper.png"

img_rgb = cv.imread(filename)
img = cv.cvtColor(img_rgb, cv.COLOR_BGR2GRAY)
template = cv.imread("one.png", cv.IMREAD_GRAYSCALE)
width, height = template.shape[::-1]

methods = ['cv.TM_CCOEFF', 'cv.TM_CCOEFF_NORMED', 'cv.TM_CCORR',
            'cv.TM_CCORR_NORMED', 'cv.TM_SQDIFF', 'cv.TM_SQDIFF_NORMED']

for meth in methods:
  img_temp = img_rgb.copy()
  method = eval(meth)
  
  res = cv.matchTemplate(img, template, method)
  min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)

  if method in [cv.TM_SQDIFF, cv.TM_SQDIFF_NORMED]:
    top_left = min_loc
  else:
    top_left = max_loc
  bottom_right = (top_left[0] + width, top_left[1] + height)

  # cv.rectangle(img_temp, top_left, bottom_right, (0, 0, 255), 2)
  cv.circle(img_temp, top_left, 20, (0, 0, 255), 2)

  cv.imshow("Display window", img_temp)
  k = cv.waitKey(0)