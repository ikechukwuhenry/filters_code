import numpy as np
import cv2
import os
# from PIL import Image
# import matplotlib.pyplot as plt

img_filename = os.path.join(os.getcwd(), 'Koala-min.jpg')

from convolution import convolve2D, processImage

def box_filter(img, kernel):
    new_img = processImage(img)
    output = convolve2D(image=new_img, kernel=kernel)
    return output

def sinc_filter(img):
    image = cv2.imread(img) 
    image = cv2.cvtColor(src=image, code=cv2.COLOR_BGR2GRAY)
    image = np.sinc(image) 
    return image


kernel = np.array([[1., 1, 1], [1, 1, 1], [1, 1, 1]])
scale_factor = 1.0 / 9
kernel = np.multiply(scale_factor, kernel)

output = box_filter(img_filename, kernel)
cv2.imwrite('box filter.jpg', output)