import cv2
import numpy as np
import os
import model_cnn

OPENCV_MAJOR_VERSION = int(cv2.__version__.split('.')[0])

def padd_symbol(img):
    max_size = max(img.shape)
    canvas = np.zeros((max_size+10, max_size+10), np.uint8)
    x = (max_size+10 - img.shape[0])//2
    y = (max_size+10 - img.shape[1])//2
    canvas[x:x+img.shape[0], y:y+img.shape[1]] = img
    return cv2.resize(canvas, (28, 28))


def inside(r1, r2):
    x1, y1, w1, h1 = r1
    x2, y2, w2, h2 = r2
    return (x1 > x2) and (y1 > y2) and (x1+w1 < x2+w2) and \
            (y1+h1 < y2+h2)

def overlape(r1, r2):
    x1, y1, w1, h1 = r1
    x2, y2, w2, h2 = r2
    if x1 > x2:
        x1, x2 = x2, x1
        w1, w2 = w2, w1
    return x2 < x1 + w1//2



img_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),'canvas.png')
img = cv2.imread(img_path, cv2.IMREAD_COLOR)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.GaussianBlur(gray, (7, 7), 0, gray)
ret, thresh = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY_INV)

if OPENCV_MAJOR_VERSION >= 4:
    # OpenCV 4 or a later version is being used.
    contours, hier = cv2.findContours(thresh, cv2.RETR_TREE,
                                      cv2.CHAIN_APPROX_SIMPLE)
else:
    # OpenCV 3 or an earlier version is being used.
    # cv2.findContours has an extra return value.
    # The extra return value is the thresholded image, which is
    # unchanged, so we can ignore it.
    _, contours, hier = cv2.findContours(thresh, cv2.RETR_TREE,
                                         cv2.CHAIN_APPROX_SIMPLE)

rectangles = []

img_h, img_w = img.shape[:2]
img_area = img_w * img_h
for c in contours:

    a = cv2.contourArea(c)
    if a >= 0.98 * img_area or a <= 0.0001 * img_area:
        continue

    r = cv2.boundingRect(c)
    is_inside = False
    for i, q in enumerate(rectangles):
        if inside(r, q):
            is_inside = True
            break
        if overlape(r, q):
            x1, y1, w1, h1 = r
            x2, y2, w2, h2 = q 
            x_new = min(x1, x2)
            y_new = min(y1, y2)
            w_new = max(x1+w1, x2+w2) - x_new
            h_new = max(y1+h1, y2+h2) - y_new
            rectangles[i] = (x_new, y_new, w_new, h_new)
            is_inside = True
            break
    if not is_inside:
        rectangles.append(r)

roi_list = []
for r in rectangles:
    x, y, w, h = r
    roi = thresh[y:y+h, x:x+w]
    roi = padd_symbol(roi)
    cv2.imshow("roi", roi)
    roi_list.append(model_cnn.normalize(roi))
    cv2.waitKey()
    cv2.rectangle(img, (x,y), (x+w, y+h), (255, 0, 0), 2)

img_vec = np.stack(roi_list, axis=0)
for each in model_cnn.predict(img_vec):
    print(np.argmax(each))