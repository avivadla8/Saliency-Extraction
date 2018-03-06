from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from skimage import io
import matplotlib.pyplot as plt
import argparse

filename = "./image_dataset/DUT-OMRON-image/DUT-OMRON-image/im005.jpg"

image = img_as_float(io.imread(filename))
numSegments = 200
segments = slic(image, n_segments = numSegments, sigma = 5,convert2lab = True)

# show the output of SLIC
fig = plt.figure("Superpixels -- %d segments" % (numSegments))
ax = fig.add_subplot(1, 1, 1)
ax.imshow(mark_boundaries(image, segments))
plt.axis("off")

# show the plots
plt.show()
