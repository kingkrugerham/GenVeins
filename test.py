from skimage import io, morphology, measure
import numpy as np
import matplotlib
from matplotlib import pyplot as plt

# Global setting for matplotlib
matplotlib.use('Agg')


def find_coords(original_im):
	"""
	Create an ndarray with tuples containing the (i, j) coordinates in the input image.
	:param original_im: Binary hand vein image from which to find coordinates.
	:return: Lists containing coordinates.
	"""
	return [(i, j) for j in range(original_im.shape[1]) for i in range(original_im.shape[0])]


def select_random_coords():
	x1 = np.random.randint(low=0, high=40)
	x2 = np.random.randint(low=0, high=40)
	y1 = np.random.randint(low=0, high=40)
	y2 = np.random.randint(low=0, high=40)
	while not ((10 < x2 - x1 < 15) and (10 < y2 - y1 < 15)):
		x1 = np.random.randint(low=0, high=40)
		x2 = np.random.randint(low=0, high=40)
		y1 = np.random.randint(low=0, high=40)
		y2 = np.random.randint(low=0, high=40)
	return (x1, x2, y1, y2)


def select_new_coords(coords):
	x1 = np.random.randint(low=1, high=49)
	x2 = np.random.randint(low=1, high=49)
	y1 = np.random.randint(low=1, high=39)
	y2 = np.random.randint(low=1, high=39)
	while not ((x2 - x1 == coords[1] - coords[0]) and (y2 - y1 == coords[3] - coords[2])):
		x1 = np.random.randint(low=1, high=49)
		x2 = np.random.randint(low=1, high=49)
		y1 = np.random.randint(low=1, high=39)
		y2 = np.random.randint(low=1, high=39)
	return (x1, x2, y1, y2)


def crop(im, coords):
	return im[coords[0]:coords[1], coords[2]:coords[3]]


def paste(im, sub_im, new_coords):
	im[new_coords[0]: new_coords[1], new_coords[2]: new_coords[3]] = sub_im
	return im


def verify_vein_spread(im):
	"""
	Verify that the random propagation algorithm introduced enough spread of the veins across the image.
	If not, image is discarded and a new one is created in its place until this function returns True.
	:param im: Image from which to calculate spread.
	:return: False if percentage of white pixels are within the given ranges.
	"""
	spread = np.count_nonzero(im)/(im.shape[0]*im.shape[1])
	if spread < 0.40 or spread > 0.80:
		return False
	return True


def add_random_objects(im):
	"""
	Add random binary objects in the unioned veins image in order to introduce more intraclass variation:
		- Select a random area in the input image
		- Copy the information in random area
		- Apply morphological operations to random area to introduce random variation
		- Paste processed random area onto another random area in the input image 
	:param im: Unioned veins image
	:return: im with added random objects
	"""
	coords = select_random_coords()
	sub_im = crop(im, coords)
	while not verify_vein_spread(sub_im):
		coords = select_random_coords()
		sub_im = crop(im, coords)
	new_coords = select_new_coords(coords)
	updated_image = paste(im, sub_im, new_coords)
	return updated_image
