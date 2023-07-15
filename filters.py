import numpy as np
import cv2
import os
from scipy import signal

gauss = np.array([[1,2,1],[2,4,2],[1,2,1]])
laplacian = np.array([[0,-1,0],[-1,4,-1],[0,-1,0]])

lapofg = signal.convolve2d(laplacian, gauss)
print(lapofg)

img_filename = os.path.join(os.getcwd(), 'Koala-min.jpg')
from convolution import convolve2D, processImage

new_img = processImage(img_filename)
output = convolve2D(image=new_img, kernel=lapofg)
cv2.imwrite('log.jpg', output)

'''
We already know that a digital image is obtained by sampling and quantizing 
the continuous signal. 
Thus if we were to interpolate a pixel value, more chances are that it resembles 
that of the neighborhood pixels and less on the distant pixels.
'''