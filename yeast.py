# Gustavo Tuani Mastrobuono
# No USP: 10734411
# SCC0251 - 2021.1
# Beer Challenge

import numpy as np
import cv2
from skimage import morphology, measure
import matplotlib.pyplot as plt

DEBUG_RECONSTRUCTION = False

# Read original image(s)
img = cv2.imread("./dataset/1Quad.jpg")

# TODO: Grid reconstruction
# Load image, grayscale, and adaptive threshold
image = img.copy()
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 57, 5)
cv2.imwrite("thresh.jpg", thresh)

# Mathematical morphology
thresh = cv2.GaussianBlur(thresh, (7,7), 0)
ret, thresh = cv2.threshold(thresh, 210, 255, 0)
kernel = np.ones((3,3), np.uint8)
morphed = cv2.erode(thresh, None, kernel, iterations=16)
morphed = cv2.morphologyEx(morphed, cv2.MORPH_CLOSE, kernel, iterations=55)
morphed = cv2.erode(morphed, None, kernel, iterations=10)
cv2.imwrite("morphed.jpg", morphed)

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
    # get the parameter of the small rectangle
    center = rect[0]
    size = rect[1]
    angle = rect[2]
    center, size = tuple(map(int, center)), tuple(map(int, size))

    # get row and col num in img
    height, width = img.shape[0], img.shape[1]
    print("width: {}, height: {}".format(width, height))

    M = cv2.getRotationMatrix2D(center, angle, 1)
    img_rot = cv2.warpAffine(img, M, (width, height))

    img_crop = cv2.getRectSubPix(img_rot, size, center)

    img_crop = cv2.rotate(img_crop, rotateCode=cv2.ROTATE_90_CLOCKWISE)
    return img_crop, img_rot

img_crop, img_rot = crop_rect(img, rect)

# Concat images
def vconcat_resize_min(im_list, interpolation=cv2.INTER_CUBIC):
    w_min = min(im.shape[1] for im in im_list)
    im_list_resize = [cv2.resize(im, (w_min, int(im.shape[0] * w_min / im.shape[1])), interpolation=interpolation)
                      for im in im_list]
    return cv2.vconcat(im_list_resize)

def hconcat_resize_min(im_list, interpolation=cv2.INTER_CUBIC):
    h_min = min(im.shape[0] for im in im_list)
    im_list_resize = [cv2.resize(im, (int(im.shape[1] * h_min / im.shape[0]), h_min), interpolation=interpolation)
                      for im in im_list]
    return cv2.hconcat(im_list_resize)

def concat_tile_resize(im_list_2d, interpolation=cv2.INTER_CUBIC):
    im_list_v = [hconcat_resize_min(im_list_h, interpolation=cv2.INTER_CUBIC) for im_list_h in im_list_2d]
    return vconcat_resize_min(im_list_v, interpolation=cv2.INTER_CUBIC)

reconstructed = concat_tile_resize([[img_crop, img_crop, img_crop, img_crop, img_crop],
                                     [img_crop, img_crop, img_crop, img_crop, img_crop],
                                     [img_crop, img_crop, img_crop, img_crop, img_crop],
                                     [img_crop, img_crop, img_crop, img_crop, img_crop],
                                     [img_crop, img_crop, img_crop, img_crop, img_crop]])

cv2.imwrite("reconstructed.jpg", reconstructed)

# TODO: Cell identification
# WIP: Contour cells
input_img = cv2.imread('./dataset/1Quad.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray, 103, 255, 1)
contours, h = cv2.findContours(thresh, cv2.RETR_TREE, 2)
for cnt in contours:
    cv2.drawContours(img, [cnt], 0, (0,0,255), 2)

cv2.imwrite("contour1.jpg", img)



# Thresholding (may be replaced by cv2 functions)
def thresholding(f, L):
    # create a new image with zeros
    f_tr = np.ones(f.shape).astype(np.uint8)
    # setting to 0 the pixels below the threshold
    f_tr[np.where(f < L)] = 0
    return f_tr

def otsu_threshold(img, max_L):
    
    M = np.product(img.shape)
    min_var = []
    hist_t,_ = np.histogram(img, bins=256, range=(0,256))
    
    img_t = thresholding(img, 0)
    
    for L in np.arange(1, max_L):
        img_ti = thresholding(img, L)
        # computing weights
        w_a = np.sum(hist_t[:L])/float(M)
        w_b = np.sum(hist_t[L:])/float(M)
        # computing variances
        sig_a = np.var(img[np.where(img_ti == 0)])
        sig_b = np.var(img[np.where(img_ti == 1)])
        
        min_var = min_var + [w_a*sig_a + w_b*sig_b]
        
    img_t = thresholding(img, np.argmin(min_var))
    
    return img_t, np.argmin(min_var)


# Get rid of background and separate cells
img = thresholding(input_img[:,:,2], 103)
#img, OL = otsu_threshold(image[:,:,2], 255)

#image[:,:,2] = np.clip(image[:,:,2], 0, 130)
#image[:,:,2] = np.clip(image[:,:,2], 0, 255)

# Erosion + dilation to get rid of healthy cells
img_di3 = morphology.binary_dilation(img, morphology.star(7)).astype(np.uint8)
img_er3 = morphology.binary_erosion(img_di3, morphology.disk(13)).astype(np.uint8)

# img_er3 = dead cells

plt.subplot(111)
plt.imshow(img_er3, cmap="gray")
plt.title("Otsu threshold") #(%d)" % (OL))
plt.show(block=False)
input('press any key')
plt.clf()

plt.figure(figsize=(15,10))


# TODO: Differentiation of living and dead cells

# Draw contours
#img_er4 = cv2.drawContours(img_er3, contours, -1, (0,255,0), 3)
#cv2.imwrite("contours.png", img_er4)

labels = measure.label(img_er3)

# TODO: Cell count
contours, hierarchy = cv2.findContours(img_er3, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
print("Approximate number of dead cells: " + str(len(contours)))

