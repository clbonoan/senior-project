import numpy as np
import matplotlib.pyplot as plt
from skimage import io, img_as_ubyte
from skimage.color import rgb2gray
from skimage.filters import laplace 
from skimage.transform import hough_line, hough_line_peaks

def load_image( location ):
    # Load the image
    image = io.imread( location )
    # If the image is color, convert it to grayscale
    if image.ndim > 2:
        image = rgb2gray( image )
    # Force the image to be treated as a [0,255] integer image
    return img_as_ubyte( image )

def hough(img):
    # Load and convert the image to grayscale
    image = load_image(img)

    # Use an edge detector like Sobel to find edges in the image
    # Reduce the image to edges of only a certain magnitude
    edges = laplace(image)
    threshold = np.absolute(edges).mean() * 0.95  # Threshold for edges, adjust as needed
    edges = edges > threshold

    # Perform the Hough Transform to find lines
    hspace, angles, distances = hough_line(edges)

    # Detect peaks in the Hough space
    h, theta, d = hough_line_peaks(hspace, angles, distances)

    # Plotting
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title('Input Image')
    axes[0].set_axis_off()

    axes[1].imshow(edges, cmap='gray')
    axes[1].set_title('Edges')
    axes[1].set_axis_off()

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.imshow(edges, cmap='gray')
    ax.set_title('Detected Lines')
    for _, angle, dist in zip(h, theta, d):
        y0 = (dist - 0 * np.cos(angle)) / np.sin(angle)
        y1 = (dist - edges.shape[1] * np.cos(angle)) / np.sin(angle)
        ax.plot((0, edges.shape[1]), (y0, y1), '-r')
    ax.set_xlim((0, edges.shape[1]))
    ax.set_ylim((edges.shape[0], 0))
    ax.set_axis_off()
    plt.tight_layout()
    plt.show()
