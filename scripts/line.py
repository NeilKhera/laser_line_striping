import math
import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

SENSOR_H = 0.00356
SENSOR_W = 0.00626
FOCAL_L  = 0.005

BASELINE = 3 * 0.0254
ANGLE    = 55

img_l = cv2.imread('left_rect.jpg')
img_r = cv2.imread('right_rect.jpg')

hsv_l    = cv2.cvtColor(img_l, cv2.COLOR_BGR2HSV)
thresh_l = cv2.inRange(hsv_l, (70, 30, 180), (100, 255, 255));
thin_l   = cv2.ximgproc.thinning(thresh_l)

cv2.imshow('left', thin_l)
cv2.waitKey(0)

rows, cols, _ = img_l.shape
points = []

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

    z = (FOCAL_L * BASELINE) / disparity
    x = x_l * BASELINE / disparity
    y = y_l * BASELINE / disparity
    points.append([x, y, z])

for i in range(len(points)):
    py = (points[i][1] * math.cos(ANGLE * math.pi / 180)) \
       - (points[i][2] * math.sin(ANGLE * math.pi / 180))
    pz = (points[i][1] * math.sin(ANGLE * math.pi / 180)) \
       + (points[i][2] * math.cos(ANGLE * math.pi / 180))

    points[i][1] = py
    points[i][2] = pz

z_vals = []
y_vals = []
x_vals = []
for p in points:
    [px, py, pz] = p
    z_vals.append(-pz)
    y_vals.append(-py)
    x_vals.append(-px)

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.set_xlim3d(-0.25, 0.25)
ax.set_ylim3d(0.15,0.65)
ax.set_zlim3d(-0.25,0.0)
ax.plot(x_vals, y_vals, z_vals)
plt.show()
