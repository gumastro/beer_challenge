# Gustavo Tuani Mastrobuono
# No USP: 10734411
# SCC0251 - 2021.1
# Beer Challenge

import numpy as np
import cv2
from skimage import morphology, measure
import matplotlib.pyplot as plt

# Read original image(s)
img = cv2.imread("./dataset/1Quad.jpg")

# TODO: Grid reconstruction
# Load image, grayscale, and adaptive threshold
image = cv2.imread('./dataset/1Quad.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 57, 5)
cv2.imwrite("thresh.jpg", thresh)

thresh=cv2.GaussianBlur(thresh,(7,7),0)
ret, thresh = cv2.threshold(thresh,210, 255, 0)
kernel = np.ones((3,3), np.uint8)
test = cv2.erode(thresh, None, kernel, iterations=16)
test = cv2.morphologyEx(test, cv2.MORPH_CLOSE, kernel, iterations=55)
test = cv2.erode(test, None, kernel, iterations=10)

cv2.imwrite("filter.jpg", test)

contours, hierarchy = cv2.findContours(image=test, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
cv2.drawContours(image=img, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=5, lineType=cv2.LINE_AA)

c = max(contours, key = cv2.contourArea)

def scale_contour(cnt, scale):
    M = cv2.moments(cnt)
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])

    cnt_norm = cnt - [cx, cy]
    cnt_scaled = cnt_norm * scale
    cnt_scaled = cnt_scaled + [cx, cy]
    cnt_scaled = cnt_scaled.astype(np.int32)

    return cnt_scaled

c = scale_contour(c, 1.15)

print(c)
x,y,w,h = cv2.boundingRect(c)

cv2.circle(img,(x,y), 8, (0,255,0), -1)
cv2.circle(img,(x+w,y), 8, (0,255,0), -1)
cv2.circle(img,(x+w,y+h), 8, (0,255,0), -1)
cv2.circle(img,(x,y+h), 8, (0,255,0), -1)
#cv2.imshow("Corners of grid", img)

# draw the biggest contour (c) in green
cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),5)

plt.imshow(img)
plt.title("Otsu") #(%d)" % (OL))
plt.show(block=False)
input('press any key')
plt.clf()




test = cv2.erode(test, None, kernel, iterations=30)
cv2.imwrite("test.jpg", test)

vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,5))
thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, vertical_kernel, iterations=9)
horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,1))
thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, horizontal_kernel, iterations=4)
cv2.imwrite("lines.jpg", thresh)

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

