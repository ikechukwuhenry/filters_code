import math
import cv2
import numpy as np
import os

def gkernel(l=3, sig=2):
    """\
    Gaussian Kernel Creator via given length and sigma
    """

    ax = np.linspace(-(l - 1) / 2., (l - 1) / 2., l)
    xx, yy = np.meshgrid(ax, ax)

    kernel = np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(sig))

    return kernel / np.sum(kernel)


print(gkernel())

filename = 'opencv_logo.jpg'
img_file= os.path.join(os.getcwd(), filename)
img = cv2.imread(img_file)
 
# Apply 3x3 and 7x7 Gaussian blur
low_sigma = cv2.GaussianBlur(img,(3,3),0)
high_sigma = cv2.GaussianBlur(img,(5,5),0)
 
# Calculate the DoG by subtracting
difference_of_gaussian = low_sigma - high_sigma

# Display the image
cv2.imshow('low sigma',low_sigma)
cv2.imshow('high sigma',high_sigma)
cv2.imshow('difference of gaussian',difference_of_gaussian)
cv2.waitKey(0)
cv2.destroyAllWindows()