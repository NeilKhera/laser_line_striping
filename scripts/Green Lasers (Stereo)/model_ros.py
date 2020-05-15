#!/usr/bin/env python

import cv2
import math
import time
import numpy as np
from scipy.spatial.transform import Rotation as R

import rospy
import message_filters
from cv_bridge import CvBridge
from sensor_msgs import point_cloud2

from std_msgs.msg import Header
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image
from sensor_msgs.msg import PointCloud2

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

l_C = np.array([[ 1035.35966,     0.     ,   653.49527],
                [    0.     ,  1035.64358,   369.82718],
                [    0.     ,     0.     ,     1.     ]])
l_D = np.array([-0.337964, 0.123943, -0.000500, -0.000057, 0.000000])
l_R = np.array([[ 0.99981329,  0.00214123,  0.0192041 ],
                [-0.00218342,  0.99999525,  0.00217637],
                [-0.01919934, -0.00221789,  0.99981322]])
l_P = np.array([[ 999.10542 ,     0.     ,   640.06171,    0.     ],
                [    0.     ,   999.10542,   365.41253,    0.     ],
                [    0.     ,     0.     ,     1.     ,    0.     ]])

r_C = np.array([[ 1030.47087,     0.     ,   664.83906],
                [    0.     ,  1029.95351,   360.74551],
                [    0.     ,     0.     ,     1.     ]])
r_D = np.array([-0.343583, 0.134452, -0.000074, -0.000862, 0.000000])
r_R = np.array([[ 0.99987644,  0.00040133,  0.01571418],
                [-0.0003668 ,  0.99999751, -0.00220018],
                [-0.01571502,  0.00219415,  0.9998741 ]])
r_P = np.array([[ 999.10542 ,     0.     ,   640.06171,  -79.98422],
                [    0.     ,   999.10542,   365.41253,    0.     ],
                [    0.     ,     0.     ,     1.     ,    0.     ]])

SENSOR_H = 0.00356
SENSOR_W = 0.00626
FOCAL_L  = 0.005
BASELINE = 3 * 0.0254
ANGLE    = 55

bridge = CvBridge()
points = []

first_t = 0
hack_t = 0
first = True

pointcloud_pub = rospy.Publisher('/points', PointCloud2, queue_size=1)

def imageCallback(left_msg, right_msg, odom_msg):
  global first, first_t, hack_t
  if first:
    first_t = int(round(time.time() * 1000))
    first = False
    
  hack_t = int(round(time.time() * 1000))

  print(hack_t)

  img_l = bridge.imgmsg_to_cv2(left_msg, desired_encoding='passthrough')
  img_r = bridge.imgmsg_to_cv2(right_msg, desired_encoding='passthrough')
  #img_l = cv2.imread('/home/nkhera/catkin_ws/src/laser_line_stripping/scripts/images/left2_rect.jpg')
  #img_r = cv2.imread('/home/nkhera/catkin_ws/src/laser_line_stripping/scripts/images/right2_rect.jpg')
  rows, cols, _ = img_l.shape

  l_map1, l_map2 = cv2.initUndistortRectifyMap(l_C, l_D, l_R, l_P, (cols, rows), cv2.CV_32FC1)
  r_map1, r_map2 = cv2.initUndistortRectifyMap(r_C, r_D, r_R, r_P, (cols, rows), cv2.CV_32FC1)
  img_l = cv2.remap(img_l, l_map1, l_map2, cv2.INTER_CUBIC)
  img_r = cv2.remap(img_r, r_map1, r_map2, cv2.INTER_CUBIC)
  rows, cols, _ = img_l.shape
  #cv2.imshow('l', img_l)
  #cv2.imshow('r', img_r)
  #cv2.waitKey(0)

  for i in range(rows):
    max_l = 0
    max_r = 0
    ind_l = 0
    ind_r = 0
    for j in range(cols):
      if img_l[i, j][1] > max_l:
        max_l = img_l[i, j][1]
        ind_l = j
      if img_r[i, j][1] > max_r:
        max_r = img_r[i, j][1]
        ind_r = j

    x_l = (float((ind_l - (cols / 2)) + 0.5) / cols) * SENSOR_W
    x_r = (float((ind_r - (cols / 2)) + 0.5) / cols) * SENSOR_W
    y_l = (float((i - (rows / 2)) + 0.5) / rows) * SENSOR_H
    y_r = (float((i - (rows / 2)) + 0.5) / rows) * SENSOR_H
    disparity = x_l - x_r

    if disparity > 0 and max_l > 150 and max_r > 150:
      z = (FOCAL_L * BASELINE) / disparity
      x = x_l * BASELINE / disparity
      y = y_l * BASELINE / disparity

      y_temp = (y * math.cos(ANGLE * math.pi / 180)) \
             - (z * math.sin(ANGLE * math.pi / 180))
      z_temp = (y * math.sin(ANGLE * math.pi / 180)) \
             + (z * math.cos(ANGLE * math.pi / 180))
      #point  = [-z_temp, -x, -y_temp]
      point = [-x, -y_temp, -z_temp]

      hack_a = (float(hack_t - first_t) / 43920) * 360
      hack_r = R.from_euler('z', -hack_a, degrees=True)
      hack_p = hack_r.apply(point)

      qx = odom_msg.pose.pose.orientation.x
      qy = odom_msg.pose.pose.orientation.y
      qz = odom_msg.pose.pose.orientation.z
      qw = odom_msg.pose.pose.orientation.w
      ro = R.from_quat([qx, qy, qz, qw])
      point_rotated = ro.apply(point)

      tx = odom_msg.pose.pose.position.x
      ty = odom_msg.pose.pose.position.y
      tz = odom_msg.pose.pose.position.z
      point_translated = point_rotated + [tx, ty, tz]

      if point_rotated[0] <= 0.6 and point_rotated[0] >= -0.6:
        if point_rotated[1] <= 0.6 and point_rotated[1] >= -0.6:
          if point_rotated[2] <= -0.12:
            points.append(hack_p)

  header = Header()
  header.frame_id = 'map'
  pc2 = point_cloud2.create_cloud_xyz32(header, points)
  pointcloud_pub.publish(pc2)
  
  #rospy.signal_shutdown('BEEP')

def run():
  rospy.init_node('line_striping_node', disable_signals=True)
  l_sub = message_filters.Subscriber('/mipi/cam0', Image)
  r_sub = message_filters.Subscriber('/mipi/cam1', Image)
  o_sub = message_filters.Subscriber('/odom', Odometry)
  
  ts = message_filters.ApproximateTimeSynchronizer([l_sub, r_sub, o_sub], 1, 0.2)
  ts.registerCallback(imageCallback)
  rospy.spin()

  #header = Header()
  #header.frame_id = 'map'
  #pc2 = point_cloud2.create_cloud_xyz32(header, points)
  #pointcloud_pub.publish(pc2)


  #z_vals = []
  #y_vals = []
  #x_vals = []
  #for p in points:
    #[px, py, pz] = p
    #if pz < -0.10:
      #z_vals.append(pz)
      #y_vals.append(py)
      #x_vals.append(px)

  #fig = plt.figure()
  #ax = fig.gca(projection='3d')
  #ax.set_xlim3d(-0.65, 0.65)
  #ax.set_ylim3d(-0.65, 0.65)
  #ax.set_zlim3d(-0.25, -0.05)
  #ax.plot(x_vals, y_vals, z_vals, 'b,')
  #plt.show()

if __name__=='__main__':
  try:
    run()
  except rospy.ROSInterruptException:
    pass
