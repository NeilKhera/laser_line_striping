import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

SENSOR_H = 0.00356
SENSOR_W = 0.00626
FOCAL_L  = 0.005
ANGLE    = 55

plane1 = (0, 0, 0, 0)
plane2 = (0, 0, 0, 0)

def get_points(plane, x_r, y_r):
  (A, B, C, D) = plane
    
  z = (-D * FOCAL_L) / ((A * x_r) + (B * y_r) + (C * FOCAL_L))
  x = x_r * z / FOCAL_L
  y = y_r * z / FOCAL_L

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

l1x1 = 0
l1y1 = 158
l1x2 = 1279
l1y2 = 152

l2x1 = 0
l2y1 = 446
l2x2 = 1279
l2y2 = 458

img = cv.imread('images/frame.png')
rows, cols, _ = img.shape
map1, map2 = cv.initUndistortRectifyMap(r_C, r_D, r_R, r_P, (cols, rows), cv.CV_32FC1)
img = cv.remap(img, map1, map2, cv.INTER_CUBIC)

red = img[:, :, 2]
dst = cv.fastNlMeansDenoising(red,None,30,7,21)
ret, thresh = cv.threshold(dst, 170, 255, cv.THRESH_BINARY)
thinned = cv.ximgproc.thinning(thresh)

points = []
for i in range(rows):
  for j in range(cols):
    if thinned[i][j] == 255:
      x_r = (float((j - (cols / 2)) + 0.5) / cols) * SENSOR_W
      y_r = (float((i - (rows / 2)) + 0.5) / rows) * SENSOR_H
      if i < 360:
        points.append(get_point(plane1, x_r, y_r))
      else:
        points.append(get_point(plane2, x_r, y_r))

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
ax.plot(x_vals, y_vals, z_vals)
plt.show()
