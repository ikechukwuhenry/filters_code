import numpy as np
import cv2
import os
from PIL import Image
# import matplotlib.pyplot as plt

img_filename = os.path.join(os.getcwd(), 'Koala-min.jpg')
color_img = np.asarray(Image.open(img_filename)) / 255


kernel = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

print(kernel)


kernel = np.flipud(kernel)      # flips the matrix vertically i.e up to down
print(kernel)
kernel = np.fliplr(kernel)      # flips the matrix horizontally i.e left to right
# kernel = np.flipud(np.fliplr(kernel))
print(kernel)

# for cross correlation/or convolution flip the matrix horizontally then vertically
kernel = np.flipud(np.fliplr(kernel))
print(kernel)

color_img = cv2.imread(img_filename)

print(color_img.shape)
cv2.imshow('image',color_img)
cv2.waitKey(0)

kernel = np.array([[1., 1, 1], [1, 1, 1], [1, 1, 1]])
scale_factor = 1.0 / 9
kernel = np.multiply(scale_factor, kernel)
print(kernel)
