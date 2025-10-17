# combines canny edge detection, local binary pattern (LBP), and shadow mask
# for texture analysis (first feature)

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# LOCAL BINARY PATTERN
def get_pixel(img, center, x, y):
    new_value = 0

    try:
        # if local neighbor pixel >= center pixel 
        if img[x][y] >= center:
            new_value = 1
    except:
        # exception where neighbor value of center pixel is null
        # i.e., values present at boundaries
        pass

    return new_value

# calculate lbp
def lbp_calculated_pixel(img, x, y): 
    center = img[x][y]

    # create array of pixels
    val_ar = []

    # top left
    val_ar.append(get_pixel(img, center, x-1, y-1))

    # top
    val_ar.append(get_pixel(img, center, x-1, y))

    # top right
    val_ar.append(get_pixel(img, center, x-1, y+1))

    # right
    val_ar.append(get_pixel(img, center, x, y+1))

    # bottom right
    val_ar.append(get_pixel(img, center, x+1, y+1))

    # bottom
    val_ar.append(get_pixel(img, center, x+1, y))

    # bottom left
    val_ar.append(get_pixel(img, center, x+1, y-1))

    # left
    val_ar.append(get_pixel(img, center, x, y-1))

    # convert binary values to decimal
    power_val = [1, 2, 4, 8, 16, 32, 64, 128]

    val = 0

    for i in range(len(val_ar)):
        val += val_ar[i] * power_val[i]

    return val


def lbp_map(gray):
    height,width = gray.shape
    out = np.zeros((height,width), np.uint8)
    for i in range(height):
        for j in range(width):
            out[i,j] = lbp_calculated_pixel(gray, i, j)
    return out

def lbp_hist(patch):
    # 8-neighbor lbp -> values 0 to 255
    hist, _ = np.histogram(patch.ravel(), bins=256, range=(0,256), density=True)
    return hist


# SHADOW MASK
# necessary for isolating shadows from dark non-shadow objexts
# goal is to focus edge detection and texture analysis only on shadow regions

