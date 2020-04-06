import math
import cv2
import numpy as np
import matplotlib.pyplot as plt

SENSOR_H = 0.00356
SENSOR_W = 0.00626
FOCAL_L  = 0.005

BASELINE = 3 * 0.0254
ANGLE    = -35

img_l = cv2.imread('left.jpg')
img_r = cv2.imread('right.jpg')

#hsv_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2HSV)
#hsv_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2HSV)
#lower_green = np.array([50, 100, 100])
#upper_green = np.array([70, 255, 255])
#thresholded_l = cv2.inRange(hsv_l, lower_green, upper_green)
#thresholded_r = cv2.inRange(hsv_r, lower_green, upper_green)
#thinning_l = cv2.ximgproc.thinning(thresholded_l)
#thinning_r = cv2.ximgproc.thinning(thresholded_r)

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

    x_l = (float(ind_l + 1) / cols) * SENSOR_W
    x_r = (float(ind_r + 1) / cols) * SENSOR_W
    y_l = ((float(i - (rows / 2)) - 0.5) / rows) * SENSOR_H
    y_r = ((float(i - (rows / 2)) - 0.5) / rows) * SENSOR_H
    disparity = x_l - x_r

    z = (FOCAL_L * BASELINE) / disparity
    x = x_l * z / FOCAL_L
    y = y_l * z / FOCAL_L
    points.append([x, y, z])

for i in range(len(points)):
    py = (points[i][1] * math.cos(ANGLE * math.pi / 180)) \
       - (points[i][2] * math.sin(ANGLE * math.pi / 180))
    pz = (points[i][1] * math.sin(ANGLE * math.pi / 180)) \
       + (points[i][2] * math.cos(ANGLE * math.pi / 180))

    points[i][1] = py
    points[i][2] = pz

#print(len(points))
z_vals = []
y_vals = []
for p in points:
    [px, py, pz] = p
    z_vals.append(pz)
    y_vals.append(-py)

plt.plot(z_vals, y_vals)
plt.show()
#cv2.imshow('left', img_l)
#cv2.imshow('right', img_r)
