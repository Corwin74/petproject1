import cv2
img_path = "/home/alex/projects/opencv/canvas.png"
img = cv2.imread(img_path, cv2.IMREAD_COLOR)
print("img shape: ", img.shape)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
print("img shape: ", gray.shape)
cv2.imshow("thresh", gray)
cv2.waitKey()
 