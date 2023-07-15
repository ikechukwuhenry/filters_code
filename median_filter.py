import cv2
import numpy as np
from skimage.util import random_noise


# cv2.medianBlur(input_image, kernel_size)
'''
This is a non-linear filtering technique. 
As clear from the name, this takes a median of all the pixels under the 
kernel area and replaces the central element with this median value. 
This is quite effective in reducing a certain type of noise (like salt-and-pepper noise) 
with considerably less edge blurring as compared to other linear filters of the same size.

Because we are taking a median, the output image will have no new pixel values 
other than that in the input image.

Note: For an even number of entries, there is more than one possible median, 
thus kernel size must be odd and greater than 1 for simplicity.
for more info and culled from:
https://theailearner.com/2019/05/06/smoothing-filters/
'''
 
# Load an image
im_arr = cv2.imread("Koala-min.jpg")
 
# Add salt and pepper noise to the image
noise_img = random_noise(im_arr, mode="s&p",amount=0.3)
noise_img = np.array(255*noise_img, dtype = 'uint8')
 
# Apply median filter
median = cv2.medianBlur(noise_img,5)
 
# Display the image
cv2.imshow('blur',noise_img)
cv2.imshow('blur1',median)
cv2.waitKey(0)
cv2.destroyAllWindows()