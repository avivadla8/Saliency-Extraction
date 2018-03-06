import numpy as np
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from skimage import io
import matplotlib.pyplot as plt
import argparse
from skimage.color import rgb2lab
from scipy.spatial.distance import cdist

def show_output(superpixels,rgb_image,mean_rgb,type):
    if type=="abstraction":
        out_image = np.zeros(rgb_image.shape)
        for superpixel in np.unique(superpixels):
            mask = superpixels == superpixel
            out_image[mask,:] = mean_rgb[superpixel,:]
        fig = plt.figure("Abstraction")
    ax = fig.add_subplot(1,1,1)
    ax.imshow(out_image)
    plt.axis("off")
    plt.show()

def apply_slic(numSegments,rgb_image):
    image = img_as_float(rgb_image)
    image = rgb2lab(image)
    segments = slic(image, n_segments = numSegments, sigma = 5,convert2lab = False)

    ## Plotting the segments
    # fig = plt.figure("Superpixels -- %d segments" % (numSegments))
    # ax = fig.add_subplot(1, 1, 1)
    # ax.imshow(mark_boundaries(rgb_image, segments))
    # plt.axis("off")
    # plt.show()
    return segments
def apply_abstraction(superpixels,rgb_image):
    rgb_image1 = img_as_float(rgb_image)
    lab_image1 = rgb2lab(rgb_image1)

    # construct position matrix
    max_y, max_x = np.array(superpixels.shape) - 1
    x = np.linspace(0, max_x, rgb_image.shape[1]) / max_x
    y = np.linspace(0, max_y, rgb_image.shape[0]) / max_y
    position = np.dstack((np.meshgrid(x, y)))
    numSegments = superpixels.max()+1

    # compute mean color and position
    mean_lab = np.zeros((numSegments, 3))
    mean_rgb = np.zeros((numSegments, 3))
    mean_position = np.zeros((numSegments, 2))
    for superpixel in np.unique(superpixels):
        mask = superpixels == superpixel
        mean_lab[superpixel, :] = lab_image1[mask, :].mean(axis=0)
        mean_rgb[superpixel, :] = rgb_image1[mask, :].mean(axis=0)
        mean_position[superpixel, :] = position[mask, :].mean(axis=0)

    return mean_rgb,mean_lab,mean_position

# filename = "./image_dataset/DUT-OMRON-image/DUT-OMRON-image/im005.jpg"
filename = "sample.jpg"
rgb_image = io.imread(filename)
print(rgb_image.shape)
superpixels = apply_slic(300,rgb_image)
print(superpixels.max()+1)
print(superpixels.shape)
mean_rgb,mean_lab,mean_position = apply_abstraction(superpixels,rgb_image)
show_output(superpixels,rgb_image,mean_rgb,"abstraction")
