import cv2
import numpy as np



img_path = "/home/alex/projects/opencv/canvas.png"
img = cv2.imread(img_path, cv2.IMREAD_COLOR)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.GaussianBlur(gray, (7, 7), 0, gray)
ret, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

cv2.imshow('img', img)
cv2.imshow("gauss", gray)
cv2.imshow('thresh', thresh)
cv2.waitKey()