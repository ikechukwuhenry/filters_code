import numpy as np
import cv2
import os
from PIL import Image
# import matplotlib.pyplot as plt

# img_filename = os.path.join(os.getcwd(), 'Koala-min.jpg')
# color_img = np.asarray(Image.open(img_filename)) / 255


kernel = np.array([[1., 1, 1], [1, 1, 1], [1, 1, 1]])
scaling_factor = 1./9
kernel = np.multiply(scaling_factor, kernel)


from scipy import linalg

rng = np.random.default_rng()

m, n = 9, 6

a = rng.standard_normal((m, n)) + 1.j*rng.standard_normal((m, n))

U, s, Vh = linalg.svd(kernel)
print(s)
print(U)
print(Vh)
print(U.shape)
k1 = np.dot(np.multiply(s[0], U[0][0]), Vh[0][0])
print(k1)

img_filename = os.path.join(os.getcwd(), 'Koala-min.jpg')

def processImage(image): 
  image = cv2.imread(image) 
#   image = cv2.cvtColor(src=image, code=cv2.COLOR_BGR2GRAY)
  img = np.sinc(image)
  return img



# img = processImage(img_filename)
# # cv2.imwrite('sinc.jpg', img)
# cv2.imshow('sinc ', img)
# cv2.waitKey(0)

# Load the image as grayscale
img = cv2.imread(img_filename, 0)
# Apply Gaussian Blur
blur = cv2.GaussianBlur(img,(3,3),0)
 
# Apply Laplacian operator in some higher datatype
laplacian = cv2.Laplacian(blur,cv2.CV_64F)

# display images
cv2.imshow('blur image', blur)
cv2.waitKey(0)
cv2.imshow('lap', laplacian)
cv2.waitKey(0)