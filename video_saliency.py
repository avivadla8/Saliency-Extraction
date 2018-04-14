import cv2
import numpy as np
import os
import sys
from main import *

filename = sys.argv[1]
video_name = sys.argv[2]

video_cap = cv2.VideoCapture(filename)
success, image = video_cap.read()
# cv2.imshow('image',image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

size = (image.shape[0],image.shape[1])
fps = 24

fourcc = cv2.VideoWriter_fourcc('D','I','V','X')
out_video = cv2.VideoWriter(video_name,fourcc,fps,size)
count = 0
while success:
	out_image = perform_saliency(image)
	out_video.write(out_image)
	success,image = video_cap.read()
	count +=1

out_video.release()