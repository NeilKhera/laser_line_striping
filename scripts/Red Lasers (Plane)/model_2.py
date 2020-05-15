#!/usr/bin/env python

import math
import glob
import numpy as np
import cv2 as cv

import rospy
from cv_bridge import CvBridge
from sensor_msgs import point_cloud2

from std_msgs.msg import Header
from sensor_msgs.msg import Image
from sensor_msgs.msg import PointCloud2

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

r_C = np.array([[ 514.41205,    0.     ,  329.83671],
                [   0.     ,  685.92876,  237.71471],
                [   0.     ,    0.     ,    1.     ]])
r_D = np.array([-0.350373, 0.158447, 0.000735, -0.000231, 0.000000])
r_R = np.array([[ 0.99999016, -0.00223406,  0.00383214],
                [ 0.00224365,  0.99999436, -0.00249788],
                [-0.00382653,  0.00250645,  0.99998954]])
r_P = np.array([[ 658.11411,    0.     ,  326.21634,  -50.15808],
                [   0.     ,  658.11411,  240.2268 ,    0.     ],
                [   0.     ,    0.     ,    1.     ,    0.     ]])

plane = (0.04493705, -0.93709475, 0.3461706, -0.16562456)

SENSOR_H = 0.00356
SENSOR_W = 0.00626
FOCAL_L  = 0.005
ANGLE    = -21.125

bridge = CvBridge()
points = []
index = 0

pointcloud_pub = rospy.Publisher('/points', PointCloud2, queue_size=1)

def get_points(plane, x_r, y_r):
  (A, B, C, D) = plane
    
  z = (-D * FOCAL_L) / ((A * x_r) + (B * y_r) + (C * FOCAL_L))
  x = x_r * z / FOCAL_L
  y = y_r * z / FOCAL_L

  return [x, y, z]

def imageCallback():
  global index
  filenames = glob.glob("test/*.jpg")
  filenames.sort()
  images = [cv.imread(img) for img in filenames]

  for img in images:
    print(index)
    rows, cols, _ = img.shape
    map1, map2 = cv.initUndistortRectifyMap(r_C, r_D, r_R, r_P, (cols, rows), cv.CV_32FC1)
    img = cv.remap(img, map1, map2, cv.INTER_CUBIC)

    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    lb = np.array([0, 60, 180])
    ub = np.array([10, 255, 255])
    ran = cv.inRange(hsv, lb, ub)

    open_kern = np.ones((9, 9), dtype=np.uint8)
    bin_y = cv.dilate(ran, open_kern)
    thinned = cv.ximgproc.thinning(bin_y)

    for i in range(rows):
      for j in range(cols):
        if thinned[i][j] == 255:
          x_r = (float((j - (cols / 2)) + 0.5) / cols) * SENSOR_W
          y_r = (float((i - (rows / 2)) + 0.5) / rows) * SENSOR_H
          p = get_points(plane, x_r, y_r)

          py = (p[1] * math.cos(ANGLE * math.pi / 180)) \
             - (p[2] * math.sin(ANGLE * math.pi / 180))
          pz = (p[1] * math.sin(ANGLE * math.pi / 180)) \
             + (p[2] * math.cos(ANGLE * math.pi / 180))
   
          po = [p[0], py + index * 0.02, 0.5 - pz]

          if po[0] <= 1 and po[0] >= -1:
            if po[2] <= 0.2 and po[2] >= -0.05:
              points.append(po)
  
    index = index + 1
    header = Header()
    header.frame_id = 'map'
    pc2 = point_cloud2.create_cloud_xyz32(header, points)
    pointcloud_pub.publish(pc2)

def run():
  rospy.init_node('line_striping_node', anonymous=True)
  imageCallback()

if __name__=='__main__':
  try:
    run()
  except rospy.ROSInterruptException:
    pass
