import numpy as np
from scipy import signal
import cv2

def gabor_kernel(size, wavelength, orientation, sigma, psi=0, gamma=1):
    """Returns a Gabor kernel with the given parameters."""
    radius = (size - 1) / 2
    x, y = np.meshgrid(np.arange(-radius, radius + 1), np.arange(-radius, radius + 1))
    
    # Rotation
    x_theta = x * np.cos(orientation) + y * np.sin(orientation)
    y_theta = -x * np.sin(orientation) + y * np.cos(orientation)
    
    # Gabor function
    gb = np.exp(-(x_theta**2 + gamma**2 * y_theta**2) / (2 * sigma**2)) * np.cos(2 * np.pi * x_theta / wavelength + psi)
    
    # Normalize
    gb -= np.mean(gb)
    gb /= np.sqrt(np.sum(gb**2))
    
    return gb

# Example usage
gabor = gabor_kernel(25, 8, np.pi/4, 4)

# Load input image
img = cv2.imread('/Users/mac/Desktop/MEDICAL_AI_PROJECTS/topology_deep_learning/Koala-min.jpg', cv2.IMREAD_GRAYSCALE)

# Check if image is loaded correctly
if img is None:
    print('Error: Could not read image')
else:
    # Define Gabor kernel parameters
    ksize = 31
    sigma = 5
    theta = np.pi / 4
    lamda = np.pi / 4
    gamma = 0.5
    psi = 0

    # Create Gabor kernel
    kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lamda, gamma, psi, ktype=cv2.CV_32F)

    # Apply Gabor filter
    filtered = cv2.filter2D(img, cv2.CV_8UC3, kernel)

    # Check if output image is empty
    if filtered is None:
        print('Error: Filtered image is empty')
    else:
        # Display input and output images
        cv2.imshow('Input image', img)
        cv2.imshow('Gabor filtered image', filtered)
        cv2.waitKey(0)

