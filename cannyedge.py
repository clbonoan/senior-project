import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

#Source: https://docs.opencv.org/3.4/da/d22/tutorial_py_canny.html

def canny(img):
    img = cv.imread(img, cv.IMREAD_GRAYSCALE)
    assert img is not None, "file could not be read, check with os.path.exists()"
    
    # threshold values to control edge sensitivity
    # lower threshold = detects weaker edges
    # higher threshold = ignores noise and weak gradients 
    edges = cv.Canny(img,100,200)

    # figure with 1 row 2 columns and activate first subplot
    # gray colormap
    plt.subplot(121),plt.imshow(img,cmap = 'gray')
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    
    # figure with 1 row 2 columns and activate second subplot
    plt.subplot(122),plt.imshow(edges,cmap = 'gray')
    plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
    plt.show()