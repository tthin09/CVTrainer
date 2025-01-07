import cv2 as cv
import math
import numpy as np
import time
from matplotlib import pyplot as plt
from utils import RED, BLUE, BLACK, WHITE, YELLOW

class Graph:
  def __init__(self):
    self.left_angles = []
    self.right_angles = []
    
  def addData(self, left, right):
    self.left_angles.append(left)
    self.right_angles.append(right)
    
  def plotGraph(self):
    plt.plot(range(len(self.left_angles)), self.left_angles, label="Left Angle")
    plt.plot(range(len(self.right_angles)), self.right_angles, label="Right Angle")
    plt.xlabel("Time")
    plt.ylabel("Angle")
    plt.title("Angle of 2 arm in comparison")
    plt.legend()
    plt.show()
    
class FPSCounter:
  def __init__(self):
    self.prev_time = time.time()
    self.fps_list = []

  def updateFPS(self):
    fps = self.getFPS()
    self.fps_list.append(fps)
    
  def getFPS(self):
    delta = time.time() - self.prev_time
    self.prev_time = time.time()
    fps = 1 / delta
    return fps
    
  def plotFPS(self):
    plt.plot(range(len(self.fps_list)), self.fps_list, label="FPS")
    plt.xlabel("Time")
    plt.ylabel("FPS")
    plt.title("FPS of program")
    plt.show()
    
  def printStats(self):
    data = np.array(self.fps_list)
    print("\nFPS info:")
    print(f"Mean: {np.mean(data)}")
    print(f"Median: {np.median(data)}")


class ImageProcessor:
  def __init__(self):
    self.curls_lm = [12, 14, 16, 11, 13, 15]
    self.curls_lines = [(14, 12), (14, 16), (13, 11), (13, 15)]
    self.joints = {}
    self.thresh_offset = 5
    self.min_thresh = 20 + self.thresh_offset
    self.max_thresh = 160 - self.thresh_offset
    self.font = cv.FONT_HERSHEY_SIMPLEX
    self.min_thresh_hist = [self.min_thresh]
    self.max_thresh_hist = [self.max_thresh]
    
    self.image = None
    self.landmarks = None
    self.width = 0
    self.height = 0
    
    self.mode = "start"
    self.is_in_rep = False
    self.rep_count = 0
    self.start_rep_time = 100
    self.end_rep_time = 100
    
    self.graph = Graph()
    self.fps_counter = FPSCounter()
    
  def loadImage(self, image, landmarks):
    self.image = image
    self.landmarks = landmarks
    self.height, self.width, temp = image.shape
    self.getJointsLocation()
    
  def getJointsLocation(self):
    self.joints = {}
    
    for index in self.curls_lm:
      lm = self.landmarks.landmark[index]
      height, width, temp = self.image.shape
      center = (int(lm.x * width), int(lm.y * height))
      key = index
      self.joints[key] = center
      
  def process(self):
    if self.mode == "start":
      self.updateThreshold()
      self.processRepCount()
      self.updateGraph()
      self.fps_counter.updateFPS()
    self.drawJoints()
    self.drawLines()
    self.drawAngleText()
    #self.drawProgressBar()
    #self.drawRepCount()
    #self.drawMode()

  def drawJoints(self):
    for key, center in self.joints.items():
      cv.circle(self.image, (center[0], center[1]), 20, (0, 0, 255), 3)
      cv.putText(self.image, str(key), (center[0], center[1] + 50), self.font, 1, RED, 3)
  
  def drawLines(self):
    for line in self.curls_lines:
      c1 = self.joints[line[0]]
      c2 = self.joints[line[1]]
      cv.line(self.image, c1, c2, (255, 255, 255), 2)
      
  def drawAngleText(self):
    left_angle, right_angle = self.getAngles()
    cv.putText(self.image, f"Left: {int(left_angle)}", (self.width - 190, 40), self.font, 1, RED, 3)
    cv.putText(self.image, f"Right: {int(right_angle)}", (self.width - 190, 100), self.font, 1, RED, 3)
    
  def drawAngleVector(self):
    p1, p2, p3, p4, p5, p6 = self.curls_lines
    # left arm
    cv.arrowedLine(self.image, p2, p1, BLUE, 3)
    cv.arrowedLine(self.image, p2, p3, BLUE, 3)
    # right arm
    cv.arrowedLine(self.image, p5, p4, BLUE, 3)
    cv.arrowedLine(self.image, p5, p6, BLUE, 3)
      
  def getAngles(self):
    vectors = []
    for line in self.curls_lines:
      c1 = self.joints[line[0]]
      c2 = self.joints[line[1]]
      vectors.append(np.array((c2[0] - c1[0], c2[1] - c1[1])))
    v1, v2, v3, v4 = vectors
    
    left_angle = math.degrees(math.acos(np.dot(v1, v2)/(np.linalg.norm(v1) * np.linalg.norm(v2))))
    right_angle = math.degrees(math.acos(np.dot(v3, v4)/(np.linalg.norm(v3) * np.linalg.norm(v4))))
    return (left_angle, right_angle)
  
  def drawProgressBar(self):
    percentage = self.getProgressPercentage()
    # draw percentage text
    text_draw_point = (self.width - 143, self.height - 420)
    cv.putText(self.image, f"{int(percentage * 100)}%", text_draw_point, self.font, 1, RED, 3)
    # draw bar
    bar_width = 100
    bar_height = 350
    border_p1 = (self.width - 50 - bar_width, self.height - 50 - bar_height)
    border_p2 = (self.width - 50, self.height - 50)
    cv.rectangle(self.image, border_p1, border_p2, BLACK, 3) # border
    bar_p1 = (border_p1[0] + 3, int(border_p2[1] - 3 - (bar_height - 6)*percentage))
    bar_p2 = (border_p2[0] - 3, border_p2[1] - 3)
    cv.rectangle(self.image, bar_p1, bar_p2, RED, -1)
  
  def getProgressPercentage(self):
    angle = min(self.getAngles())
    percentage = 1 - ((angle - self.min_thresh) / (self.max_thresh - self.min_thresh))
    # map to 0% -> 100%
    percentage = min(1, max(0, percentage))
    
    return percentage
    
  def drawRepCount(self):
    bg_p1 = (30, self.height - 72)
    bg_p2 = (200, self.height - 30)
    cv.rectangle(self.image, bg_p1, bg_p2, BLACK, -1)
    cv.putText(self.image, f"Count: {self.rep_count}", (40, self.height - 40), self.font, 1, YELLOW, 2)
  
  def processRepCount(self):
    percentage = self.getProgressPercentage()
    if percentage >= 0.1 and not self.is_in_rep:
      self.is_in_rep = True
      self.start_rep_time = time.time()
    if percentage <= 0.03 and self.is_in_rep:
      self.end_rep_time = time.time()
      delta = self.end_rep_time - self.start_rep_time
      print(f"Rep time: {delta}")
      if delta >= 2:
        self.rep_count += 1
        self.is_in_rep = False
      
  def updateThreshold(self):
    # only update on first 2 rep
    if self.rep_count > 2:
      return
    left_angle, right_angle = self.getAngles()
    angle = min(left_angle, right_angle)
    is_updated = False
    if angle + self.thresh_offset < self.min_thresh:
      print(f"Update min_thresh from {self.min_thresh} to {angle + self.thresh_offset}")
      self.min_thresh = max(20, angle + self.thresh_offset)
      is_updated = True
    if angle - self.thresh_offset > self.max_thresh:
      print(f"Update max_thresh from {self.max_thresh} to {angle - self.thresh_offset}")
      self.max_thresh = min(160, angle - self.thresh_offset)
      is_updated = True
    if is_updated:
      self.min_thresh_hist.append(self.min_thresh)
      self.max_thresh_hist.append(self.max_thresh)
      
  def updateGraph(self):
    left_angle, right_angle = self.getAngles()
    self.graph.addData(left_angle, right_angle)
    
  def plotGraph(self):
    self.graph.plotGraph()
    
  def toggleMode(self):
    if self.mode == "stop":
      self.mode = "start"
    else:
      self.mode = "stop"
      
  def drawMode(self):
    cv.rectangle(self.image, (0, 0), (120, 50), BLACK, -1)
    cv.putText(self.image, self.mode.capitalize(), (20, 35), self.font, 1, YELLOW, 2)
  
  def plotThresholdAdjust(self):
    plt.plot(range(len(self.min_thresh_hist)), self.min_thresh_hist, label="Min threshold")
    plt.xlabel("Time")
    plt.ylabel("Threshold")
    plt.title("Min threshold adjustment")
    plt.show()
    plt.plot(range(len(self.max_thresh_hist)), self.max_thresh_hist, label="Max threshold", color="orange")
    plt.xlabel("Time")
    plt.ylabel("Threshold")
    plt.title("Max threshold adjustment")
    plt.show()
  
  def end(self):
    self.plotThresholdAdjust()
    self.fps_counter.printStats()
    self.fps_counter.plotFPS()