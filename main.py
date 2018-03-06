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
    if type=="uniqueness":
        out_image = np.zeros(rgb_image.shape)
        for superpixel in np.unique(superpixels):
            mask = superpixels == superpixel
            out_image[mask,:] = mean_rgb[superpixel]
        fig = plt.figure("uniqueness")
    if type=="distribution":
        out_image = np.zeros(rgb_image.shape)
        for superpixel in np.unique(superpixels):
            mask = superpixels == superpixel
            out_image[mask,:] = mean_rgb[superpixel]
        fig = plt.figure("distribution")
    if type=="final_saliency":
        out_image = np.zeros(rgb_image.shape)
        for superpixel in np.unique(superpixels):
            mask = superpixels == superpixel
            out_image[mask,:] = mean_rgb[superpixel]
        fig = plt.figure("final_saliency")
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

    lab_image1 = (lab_image1 + np.array([0, 128, 128])) / np.array([100, 255, 255])

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

def apply_uniqueness(superpixels,mean_lab,mean_position):
    weight = np.exp(-cdist(mean_position,mean_position) ** 2/(2*0.25*0.25))
    weight = weight/(weight.sum(axis=1)[:,None])

    uniqueness = (cdist(mean_lab,mean_lab) ** 2 * weight).sum(axis=1)

    #normalize
    return (uniqueness - uniqueness.min())/(uniqueness.max()-uniqueness.min() + 1e-13)

def apply_distribution(superpixels,mean_lab,mean_position):
    weight = np.exp(-cdist(mean_lab,mean_lab) ** 2/(2*20.0*20.0))
    weight = weight/(weight.sum(axis=1)[:,None])

    print weight.shape
    print mean_position.shape
    weighted_mean = np.dot(weight,mean_position)
    print weighted_mean.shape
    #distribution = (cdist(mean_position,weighted_mean) ** 2 * weight).sum(axis=1)
    distribution = np.einsum('ij,ji->i', weight, cdist(mean_position, weighted_mean) ** 2)
    print distribution.shape
    return (distribution - distribution.min())/(distribution.max()-distribution.min() + 1e-13)

def apply_saliency(uniqueness,distribution,mean_lab,mean_position):
    saliency = uniqueness * np.exp(-6.0*distribution)

    weight = np.exp(-0.5 * (0.033 * cdist(mean_lab, mean_lab) ** 2 + 0.033 * cdist(mean_position, mean_position) ** 2))
    weight = weight/(weight.sum(axis=1)[:,None])

    weighted_saliency = np.dot(weight,saliency)
    return (weighted_saliency - weighted_saliency.min())/(weighted_saliency.max()-weighted_saliency.min() + 1e-13)

# filename = "./image_dataset/DUT-OMRON-image/DUT-OMRON-image/im005.jpg"
filename = "sample.jpg"
rgb_image = io.imread(filename)
print(rgb_image.shape)
superpixels = apply_slic(500,rgb_image)
print(superpixels.max()+1)
print(superpixels.shape)
mean_rgb,mean_lab,mean_position = apply_abstraction(superpixels,rgb_image)
#show_output(superpixels,rgb_image,mean_rgb,"abstraction")

uniqueness = apply_uniqueness(superpixels,mean_lab,mean_position)
show_output(superpixels,rgb_image,uniqueness,"uniqueness")

distribution = apply_distribution(superpixels,mean_lab,mean_position)
#show_output(superpixels,rgb_image,distribution,"distribution")

final_saliency = apply_saliency(uniqueness,distribution,mean_lab,mean_position)
show_output(superpixels,rgb_image,distribution,"final_saliency")
