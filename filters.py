import numpy as np
import cv2
import os
from scipy import signal

gauss = np.array([[1,2,1],[2,4,2],[1,2,1]])
laplacian = np.array([[0,-1,0],[-1,4,-1],[0,-1,0]])

lapofg = signal.convolve2d(laplacian, gauss)
print(lapofg)

img_filename = os.path.join(os.getcwd(), 'Koala-min.jpg')
# image = cv2.imread(img_filename) 
# image = cv2.cvtColor(src=image, code=cv2.COLOR_BGR2GRAY)
# output = signal.convolve2d(image, lapofg)
# cv2.imshow('test', output)
# cv2.waitKey(0)
from convolution import convolve2D, processImage

new_img = processImage(img_filename)
output = convolve2D(image=new_img, kernel=lapofg)
cv2.imwrite('log.jpg', output)