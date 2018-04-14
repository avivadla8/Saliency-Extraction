import numpy as np
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from skimage import io
import matplotlib.pyplot as plt
import argparse
from skimage.color import rgb2lab
from scipy.spatial.distance import cdist
from skimage.filters import threshold_otsu, threshold_local
from sys import argv

from slic_segmentation import *

import cv2

def show_output(superpixels,image,mean_rgb,type):
    if type=="abstraction":
        out_image = np.zeros(image.shape)
        for superpixel in np.unique(superpixels):
            mask = superpixels == superpixel
            out_image[mask,:] = mean_rgb[superpixel,:]
        # io.imsave('abstraction_'+str(argv[1]), (out_image * 255).astype('uint8'))
        fig = plt.figure("Abstraction")
    if type=="uniqueness":
        out_image = np.zeros(image.shape)
        for superpixel in np.unique(superpixels):
            mask = superpixels == superpixel
            out_image[mask,:] = mean_rgb[superpixel]
        # io.imsave('uniqueness_'+str(argv[1]), (out_image * 255).astype('uint8'))
        fig = plt.figure("uniqueness")
    if type=="distribution":
        out_image = np.zeros(image.shape)
        for superpixel in np.unique(superpixels):
            mask = superpixels == superpixel
            out_image[mask,:] = mean_rgb[superpixel]
        # io.imsave('distribution_'+str(argv[1]), (out_image * 255).astype('uint8'))

        fig = plt.figure("distribution")
    if type=="final_saliency":
        out_image = np.zeros(image.shape)
        for superpixel in np.unique(superpixels):
            mask = superpixels == superpixel
            out_image[mask,:] = mean_rgb[superpixel]
        # io.imsave('saliency_'+str(argv[1]), (out_image * 255).astype('uint8'))
        fig = plt.figure("final_saliency")

    if type=="Binary Mask":
        out_image = image
        fig = plt.figure("Binary Mask")

    ax = fig.add_subplot(1,1,1)
    ax.imshow(out_image)
    plt.axis("off")
    plt.show()
    return out_image

def apply_slic(numSegments,rgb_image):
    image = img_as_float(rgb_image)
    # image = rgb2lab(image)
    # segments = slic(image, n_segments = numSegments, sigma = 5,convert2lab = False)

    p = SLIC(rgb_image, numSegments, 40)
    segments = p.iterate_times(2)
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

    weighted_mean = np.dot(weight,mean_position)
    # distribution = (cdist(mean_position,weighted_mean) ** 2 * weight).sum(axis=1)
    distribution = np.einsum('ij,ji->i', weight, cdist(mean_position, weighted_mean) ** 2)
    return (distribution - distribution.min())/(distribution.max()-distribution.min() + 1e-13)

def apply_saliency(uniqueness,distribution,mean_lab,mean_position):
    saliency = uniqueness * np.exp(-6.0*distribution)

    weight = np.exp(-0.5 * (0.033 * cdist(mean_lab, mean_lab) ** 2 + 0.033 * cdist(mean_position, mean_position) ** 2))
    weight = weight/(weight.sum(axis=1)[:,None])

    weighted_saliency = np.dot(weight,saliency)
    return (weighted_saliency - weighted_saliency.min())/(weighted_saliency.max()-weighted_saliency.min() + 1e-13)

def perform_saliency(rgb_image):
    superpixels = apply_slic(500,rgb_image)
    # print(superpixels.max()+1)
    # print(superpixels.shape)
    mean_rgb,mean_lab,mean_position = apply_abstraction(superpixels,rgb_image)
    show_output(superpixels,rgb_image,mean_rgb,"abstraction")

    uniqueness = apply_uniqueness(superpixels,mean_lab,mean_position)
    show_output(superpixels,rgb_image,uniqueness,"uniqueness")

    distribution = apply_distribution(superpixels,mean_lab,mean_position)
    show_output(superpixels,rgb_image,1-distribution,"distribution")

    final_saliency = apply_saliency(uniqueness,distribution,mean_lab,mean_position)
    out = show_output(superpixels,rgb_image,final_saliency,"final_saliency")

    global_thresh = threshold_otsu(out[:,:,0])
    binary_global = out[:,:,0] > global_thresh
    final = np.zeros(out.shape)
    final[:,:,0] = binary_global
    final[:,:,1] = binary_global
    final[:,:,2] = binary_global
    show_output(superpixels,final,global_thresh,"Binary Mask")

    out = out.astype(np.uint8)
    img = cv2.medianBlur(out[:,:,0],5)
    img = out[:,:,0]
    th3 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
    final = np.zeros(out.shape)
    final[:,:,0] = th3
    final[:,:,1] = th3
    final[:,:,2] = th3
    final = final.astype(np.uint8)
    show_output(superpixels,final,final,"Binary Mask")

    bgdModel = np.zeros((1,65),np.float64)
    fgdModel = np.zeros((1,65),np.float64)

    mask = np.zeros(rgb_image.shape[:2],np.uint8)
    mask[binary_global == False] = 0
    mask[binary_global == True] = 1

    mask, bgdModel, fgdModel = cv2.grabCut(rgb_image,mask,None,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_MASK)
    mask = np.where((mask==2)|(mask==0),0,1).astype('uint8')
    fin_img = rgb_image*mask[:,:,np.newaxis]
    plt.imshow(fin_img),plt.show()


if __name__=='__main__':
    # filename = "./image_dataset/DUT-OMRON-image/DUT-OMRON-image/im005.jpg"
    filename = str(argv[1])
    rgb_image = io.imread(filename)
    perform_saliency(rgb_image)
