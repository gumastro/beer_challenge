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

print(img_er3)
labels = measure.label(img_er3)
print(labels)

# TODO: Cell count
contours, hierarchy = cv2.findContours(img_er3, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
print("Approximate number of dead cells: " + str(len(contours)))

