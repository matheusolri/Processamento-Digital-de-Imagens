from __future__ import print_function
from __future__ import division
import cv2 as cv
import numpy as np
import argparse

def invert(img, name):
    img = abs(255 - img)
    cv.imwrite(name, img)

# carregando img
parser = argparse.ArgumentParser(description='Code for Histogram Equalization.')
parser.add_argument('--input', help='Path to input image.', default='fazenda.jpg')
args = parser.parse_args()
src = cv.imread(cv.samples.findFile(args.input))
dst= cv.imread(cv.samples.findFile(args.input))
if src is None:
    print('Could not open or find the image:', args.input)
    exit(0)

# escala de cinza na img
src = cv.cvtColor(src, cv.COLOR_BGR2GRAY)

# Equalizando Histograma
img_array = np.asarray(src)
histogram_array = np.bincount(img_array.flatten(), minlength=256)
num_pixels = np.sum(histogram_array)
histogram_array = histogram_array/num_pixels
chistogram_array = np.cumsum(histogram_array)
transform_map = np.floor(255 * chistogram_array).astype(np.uint8)
img_list = list(img_array.flatten())
eq_img_list = [transform_map[p] for p in img_list]
eq_img_array = np.reshape(np.asarray(eq_img_list), img_array.shape)
src = eq_img_array

# show img
cv.imshow('Equalized Image', src)
cv.imshow('Source image', dst)
cv.waitKey()