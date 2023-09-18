# import the packages
import cv2
import matplotlib.pyplot as plt
import numpy as np


path = "hologram/F2.3.jpg"
img = cv2.imread(path)
print("shape of original image")
print(img.shape)

# conversion of image to grayscale
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
print("shape of gray image")
print(imgGray.shape)
# conversion of grayscale image to blurred image
img1 = cv2.medianBlur(imgGray, 5)


# imgResize = cv2.resize(img,(500,491))

# binarize the image
binr = cv2.adaptiveThreshold(img1, 255, cv2.ADAPTIVE_THRESH_MEAN_C,\
            cv2.THRESH_BINARY, 11, 2)
  
# define the kernel
kernel = np.ones((3, 3), np.uint8)

  
# First perform erosion then dilate the image
erosion = cv2.erode(binr, kernel, iterations=1)
# dilate the image
dilation = cv2.dilate(erosion, kernel, iterations=1)

# Finding if the image is bright or dark
meanpercent = np.mean(dilation) * 100 / 255
if meanpercent < 50:
    dark = 100-meanpercent
    print("-----------------------------------------------")
    print("The image is {0}% dark".format(dark))     
    print("and is {0}% bright".format(meanpercent))
    print("-----------------------------------------------")
else:
    dark = 100-meanpercent
    print("The image is {0}% bright".format(meanpercent))     
    print("and is {0}% dark".format(dark))


# Finding contours in image
# contours, hierarchy = cv2.findContours(image=dilation, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)
contours, hierarchy = cv2.findContours(image=dilation, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)

no_of_cont = []
for cnt in contours:
    no_of_cont.append(cnt)

print("Number of contours in the image", len(no_of_cont))
print("-----------------------------------------------")
                                      
# draw contours on the original image
image_copy = img.copy()
cv2.drawContours(image=image_copy, contours=contours, 
                 contourIdx=-1, color=(0, 255, 0), thickness=1, lineType=cv2.LINE_AA)


# print the output
cv2.imshow('Org Image', img)
cv2.imshow('Gray Image', imgGray)
cv2.imshow('Blur Image', img1)
cv2.imshow('Threshold', binr)
cv2.imshow('erosion', erosion)

cv2.imshow('dilation', dilation)
cv2.imshow('None approximation', image_copy)
# plt.imshow(erosion, cmap='gray')
cv2.waitKey(0)
