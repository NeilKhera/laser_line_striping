#!/usr/bin/env python

import cv2
import math
import numpy as np

import rospy
from cv_bridge import CvBridge
from sensor_msgs import point_cloud2

from std_msgs.msg import Header
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image
from sensor_msgs.msg import PointCloud2

SENSOR_HEIGHT  = 0.00356
SENSOR_WIDTH   = 0.00626
FOCAL_LENGTH   = 0.005
BASELINE       = 3 * 0.0254
CAMERA_ANGLE   = -35

LINE_POINTS    = ((0, 639), (719, 639))
WINDOW_SIZE    = 10
PLANE_EQUATION = (0.99978033, -0.00388811, 0.02059571, -0.03468802)

bridge    = CvBridge()
odom_msg  = Odometry()
left_msg  = Image()
right_msg = Image()

pointcloud_pub = rospy.Publisher('/points', PointCloud2, queue_size=1)

def odomCallback(msg):
  odom_msg = msg

def leftCallback(msg):
  left_msg = msg

def rightCallback(msg):
  right_msg = msg

  img_l = bridge.imgmsg_to_cv2(left_msg, desired_encodings='passthrough')
  img_r = bridge.imgmsg_to_cv2(right_msg, desired_encodings='passthrough')
  rows, cols, _ = img_l.shape
  points = []

  ind_l = LINE_POINTS[0][1]
  ind_r = LINE_POINTS[0][1]
  for i in range(rows):
    max_l = -1
    max_r = -1
    for j in range(WINDOW_SIZE):
      sl = ind_l - (WINDOW_SIZE / 2) + j
      sr = ind_r - (WINDOW_SIZE / 2) + j
      if img_l[i, sl][1] > max_l:
        max_l = img_l[i, sl][1]
        ind_l = sl
      if img_r[i, sr][1] > max_r:
        max_r = img_r[i, sr][1]
        ind_r = sr

    if max_l == -1:
      ratio = (float(i + 1) / rows)
      ind_l = int(((1.0 - t) * LINE_POINTS[0][1]) + t * LINE_POINTS[1][1])
      for j in range(WINDOW_SIZE):
        sl = ind_l - (WINDOW_SIZE / 2) + j
        if img_l[i, sl][1] > max_l:
          max_l = img_l[i, sl][1]
          ind_l = sl

    if max_r == -1:
      ratio = (float(i + 1) / rows)
      ind_r = int(((1.0 - t) * LINE_POINTS[0][1]) + t * LINE_POINTS[1][1])
      for j in range(WINDOW_SIZE):
        sr = ind_r - (WINDOW_SIZE / 2) + j
        if img_r[i, sr][1] > max_r:
          max_r = img_r[i, sr][1]
          ind_r = sr

    x_l = (float(ind_l + 1) / cols) * SENSOR_W
    x_r = (float(ind_r + 1) / cols) * SENSOR_W
    y_l = ((float(i - (rows / 2)) - 0.5) / rows) * SENSOR_H
    y_r = ((float(i - (rows / 2)) - 0.5) / rows) * SENSOR_H

    z = (-PLANE_EQUATION[3] * FOCAL_LENGTH) / ((PLANE_EQUATION[0] * x_r) \
        + (PLANE_EQUATION[1] * y_r) + (PLANE_EQUATiON[2] * FOCAL_LENGTH))
    x = (x_r * z) / FOCAL_LENGTH
    y = (y_r * z) / FOCAL_LENGTH

    px = (points[i][1] * math.sin(ANGLE * math.pi / 180)) \
       + (points[i][2] * math.cos(ANGLE * math.pi / 180))
    pz = (points[i][1] * math.cos(ANGLE * math.pi / 180)) \
       - (points[i][2] * math.sin(ANGLE * math.pi / 180))
    py = x

    points.append([px, py, pz])
  
  header = Header()
  header.frame_id = "map"
  pc2 = point_cloud2.create_cloud_xyz32(header, points)
  pointcloud_pub.publish(pc2)

def run():
  rospy.init_node('line_striping_node', anonymous=True)
  rospy.Subscriber('/odom', Odometry, odomCallback)
  rospy.Subscriber('/mipi/cam0', Image, leftCallback)
  rospy.Subscriber('/mipi/cam1', Image, rightCallback)
  rospy.spin()

if __name__=='__main__':
  try:
    run()
  except rospy.ROSInterruptException:
    pass
