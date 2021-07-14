# Gustavo Tuani Mastrobuono
# No USP: 10734411
# SCC0251 - 2021.1
# Beer Challenge

import numpy as np
import cv2

DEBUG_RECONSTRUCTION = False
DEBUG_IDENTIFICATION = True

# Read original image(s)
# TODO: Read whole dataset
img = cv2.imread("./dataset/1Quad.jpg")

# Grid reconstruction
# Load image, grayscale, and adaptive threshold
image = img.copy()
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 57, 5)
if (DEBUG_RECONSTRUCTION): cv2.imwrite("thresh.jpg", thresh)

# Mathematical morphology
thresh = cv2.GaussianBlur(thresh, (7,7), 0)
ret, thresh = cv2.threshold(thresh, 210, 255, 0)
kernel = np.ones((3,3), np.uint8)
morphed = cv2.erode(thresh, None, kernel, iterations=16)
morphed = cv2.morphologyEx(morphed, cv2.MORPH_CLOSE, kernel, iterations=55)
morphed = cv2.erode(morphed, None, kernel, iterations=10)
if (DEBUG_RECONSTRUCTION): cv2.imwrite("morphed.jpg", morphed)

# Find all contours
contours, hierarchy = cv2.findContours(morphed, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
if (DEBUG_RECONSTRUCTION): cv2.drawContours(image=img, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=5, lineType=cv2.LINE_AA)

# Get biggest contour (inner part of Neubauer chamber)
c = max(contours, key = cv2.contourArea)
if (DEBUG_RECONSTRUCTION): cv2.drawContours(image=img, contours=c, contourIdx=-1, color=(0, 0, 255), thickness=8, lineType=cv2.LINE_AA)

def scale_contour(cnt, scale):
    M = cv2.moments(cnt)
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])

    cnt_norm = cnt - [cx, cy]
    cnt_scaled = cnt_norm * scale
    cnt_scaled = cnt_scaled + [cx, cy]
    cnt_scaled = cnt_scaled.astype(np.int32)

    return cnt_scaled

# Resize contour to match real image
c = scale_contour(c, 1.16)

# Get corners of contour
rect = cv2.minAreaRect(c)
box = cv2.boxPoints(rect)
box = np.int0(box)
if (DEBUG_RECONSTRUCTION): cv2.drawContours(img, [box], 0, (255, 0, 0), 6)

def crop_rect(img, rect):
    center = rect[0]
    size = rect[1]
    angle = rect[2]
    center, size = tuple(map(int, center)), tuple(map(int, size))

    height, width = img.shape[0], img.shape[1]
    if (DEBUG_RECONSTRUCTION): print("width: {}, height: {}".format(width, height))

    M = cv2.getRotationMatrix2D(center, angle, 1)
    img_rot = cv2.warpAffine(img, M, (width, height))

    img_crop = cv2.getRectSubPix(img_rot, size, center)

    img_crop = cv2.rotate(img_crop, rotateCode=cv2.ROTATE_90_CLOCKWISE)

    return img_crop, img_rot

# Rotate and crop images
img_crop, img_rot = crop_rect(img, rect)

# Concat images
def vertical_concat(im_list, interpolation=cv2.INTER_CUBIC):
    w_min = min(im.shape[1] for im in im_list)
    im_list_resize = [cv2.resize(im, (w_min, int(im.shape[0] * w_min / im.shape[1])), interpolation=interpolation) for im in im_list]
    return cv2.vconcat(im_list_resize)

def horizontal_concat(im_list, interpolation=cv2.INTER_CUBIC):
    h_min = min(im.shape[0] for im in im_list)
    im_list_resize = [cv2.resize(im, (int(im.shape[1] * h_min / im.shape[0]), h_min), interpolation=interpolation) for im in im_list]
    return cv2.hconcat(im_list_resize)

def concat(im_list_2d, interpolation=cv2.INTER_CUBIC):
    im_list_v = [horizontal_concat(im_list_h, interpolation=cv2.INTER_CUBIC) for im_list_h in im_list_2d]
    return vertical_concat(im_list_v, interpolation=cv2.INTER_CUBIC)

reconstructed = concat([[img_crop, img_crop, img_crop, img_crop, img_crop],
                        [img_crop, img_crop, img_crop, img_crop, img_crop],
                        [img_crop, img_crop, img_crop, img_crop, img_crop],
                        [img_crop, img_crop, img_crop, img_crop, img_crop],
                        [img_crop, img_crop, img_crop, img_crop, img_crop]])

if (DEBUG_RECONSTRUCTION): cv2.imwrite("reconstructed.jpg", reconstructed)

# Cell identification
COLOR_NAMES = ["red", "orange", "yellow", "green", "cyan", "blue", "purple", "red "]

COLOR_RANGES_HSV = {
    "red": [(0, 50, 10), (10, 255, 255)],
    "orange": [(10, 50, 10), (25, 255, 255)],
    "yellow": [(25, 50, 10), (35, 255, 255)],
    "green": [(35, 50, 10), (80, 255, 255)],
    "cyan": [(80, 80, 10), (100, 255, 255)],
    "blue": [(100, 50, 10), (130, 255, 255)],
    "purple": [(130, 50, 10), (170, 255, 255)],
    "red ": [(170, 50, 10), (180, 255, 255)]
}

def getMask(frame, color):
    blurredFrame = cv2.GaussianBlur(frame, (3, 3), 0)
    hsvFrame = cv2.cvtColor(blurredFrame, cv2.COLOR_BGR2HSV)

    colorRange = COLOR_RANGES_HSV[color]
    lower = np.array(colorRange[0])
    upper = np.array(colorRange[1])

    colorMask = cv2.inRange(hsvFrame, lower, upper)
    colorMask = cv2.bitwise_and(blurredFrame, blurredFrame, mask=colorMask)

    return colorMask

def getDominantColor(roi):
    roi = np.float32(roi)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 4
    ret, label, center = cv2.kmeans(roi, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape(roi.shape)

    pixelsPerColor = []
    for color in COLOR_NAMES:
        mask = getMask(res2, color)
        greyMask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        count = cv2.countNonZero(greyMask)
        pixelsPerColor.append(count)

    return COLOR_NAMES[pixelsPerColor.index(max(pixelsPerColor))]

def getROI(frame, x, y, r):
    return frame[int(y-r/2):int(y+r/2), int(x-r/2):int(x+r/2)]
  
# Convert to grayscale, blur using 3x3 kernel and apply Hough transform on the blurred image
gray = cv2.cvtColor(reconstructed, cv2.COLOR_BGR2GRAY)  
gray_blurred = cv2.blur(gray, (3, 3))
if (DEBUG_IDENTIFICATION): cv2.imwrite("blurred.jpg", gray_blurred)
detected_circles = cv2.HoughCircles(gray_blurred, cv2.HOUGH_GRADIENT, 1, 25, param1 = 50, param2 = 23, minRadius = 1, maxRadius = 40)
  
count_total = 0
count_dead = 0
if detected_circles is not None:

    detected_circles = np.uint16(np.around(detected_circles))
    
    for pt in detected_circles[0, :]:
        # Cell count
        count_total += 1

        roi = getROI(reconstructed, pt[0], pt[1], pt[2])
        color = getDominantColor(roi)

        # Differentiation of living and dead cells
        if (color == "cyan" or color == "blue"):
            # Dead cell count
            count_dead += 1
            cv2.circle(reconstructed, (pt[0], pt[1]), pt[2], (0, 0, 255), 2)
            cv2.circle(reconstructed, (pt[0], pt[1]), 2, (0, 0, 255), 5)
            cv2.putText(reconstructed, "dead", (int(pt[0] + 40), int(pt[1] + 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))
        else:
            cv2.circle(reconstructed, (pt[0], pt[1]), pt[2], (0, 255, 0), 2)
            cv2.circle(reconstructed, (pt[0], pt[1]), 2, (0, 255, 0), 5)
            cv2.putText(reconstructed, "alive", (int(pt[0] + 40), int(pt[1] + 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))

if (DEBUG_IDENTIFICATION): cv2.imwrite("identification.jpg", reconstructed)

print("%d %d %d %.1f" % (count_total, count_total-count_dead, count_dead, (count_total-count_dead)*100/count_total))

