# System imports
import os

# Third party imports
from mahotas import labeled as lb
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
from scipy.ndimage import measurements, filters, morphology
from scipy.linalg import basic as ln
from skimage.io	import _io
from skimage.exposure import equalize_adapthist
from skimage.filters import threshold_local, threshold_otsu, gaussian
from skimage.morphology import convex_hull_image, selem, grey, _skeletonize, reconstruction, binary
from skimage.transform import resize, _warps
from skimage.draw import line
from more_itertools import pairwise

from torch import combinations

# Global setting for matplotlib
matplotlib.use('Agg')

def make_output_dir(loc):
	"""
	Create output directory if doesn't exist.
	:param loc: Output directory to check or create.
	:return: None
	"""
	if not os.path.exists(loc):
		os.makedirs(loc)


def generate_seed_images(seed_output_loc, num_ims):
	"""
	Generate binary image containing seed points to build connected hand vein-like structures from.
	The protocol is as follows:
		- Generate a black image of shape (50, 40)
		- Randomly select number of seed points between 5 and 10.
		- Randomly select a coordinate in the black image and set to 1. Repeat for number of seed points
	:param seed_output_loc: Location to save seed images.
	:param num_ims: Number of seed images to create.
	:return: None
	"""
	make_output_dir(seed_output_loc)
	for _ in range(num_ims):
		new_im = np.zeros((50, 40))
		num_seeds = np.random.randint(low=5, high=10)
		for _ in range(0, num_seeds):
			random_coord = [np.random.randint(low=0, high=50), np.random.randint(low=0, high=40)]
			new_im[random_coord[0], random_coord[1]] = 1
		plt.imsave(seed_output_loc + str(np.random.randint(1,88888))+'.png', new_im, cmap=plt.get_cmap('gray'))


def create_skeleton(seed_output_loc, skel_output_loc):
	make_output_dir(skel_output_loc)
	seed_ims = os.listdir(seed_output_loc)
	
	for im_name in seed_ims:
		im = _io.imread(seed_output_loc + im_name, as_gray=True)
		point_coords = []
		for i in range(im.shape[0]):
			for j in range(im.shape[1]):
				if im[i, j] == 1:
					point_coords.append((i, j))
		# point_coords = sorted(point_coords)
		point_pairs = list(pairwise(point_coords))
		for point_pair in point_pairs:
			rr, cc = line(point_pair[0][0], point_pair[0][1], point_pair[1][0], point_pair[1][1])
			im[rr, cc] = 1
		plt.imsave(skel_output_loc + im_name, im, cmap=plt.get_cmap('gray'))


if __name__ == '__main__':
	root_output_loc = 'C:/Users/User/Desktop/PhD_Files/Output/GANs/Struct/'
	# generate_seed_images(root_output_loc + 'Seed/', num_ims=100)
	create_skeleton(root_output_loc + 'Seed/', root_output_loc + 'Skel/')


def generate_seed_images(root_output_dir, num_ims):
	"""
	Generate binary image containing seed points to build connected hand vein-like structures from.
	The protocol is as follows:
		- Generate a black image of shape (50, 40)
		- Randomly select number of seed points between 5 and 10.
		- Randomly select a coordinate in the black image and set to 1. Repeat for number of seed points
	:param seed_output_loc: Location to save seed images.
	:param num_ims: Number of seed images to create.
	:return: None
	"""
	make_output_dir(root_output_dir + 'Seed/')
	make_output_dir(root_output_dir + 'LineSeed/')
	for i in range(num_ims):
		new_im = np.zeros((50, 40))
		origin_y, origin_x = np.random.randint(low=2, high=5), np.random.randint(low=2, high=38)
		end_y, end_x = np.random.randint(low=35, high=48), np.random.randint(low=2, high=38)
		while abs(origin_x - end_x) <= 10:
			end_x = np.random.randint(low=2, high=38)
		new_im[origin_y, origin_x] = 1
		new_im[end_y, end_x] = 1
		num_seeds = np.random.randint(low=10, high=20)
		y_dist_between_points = np.abs((end_y - origin_y)/num_seeds)

		# Make sure low <= high for random.randint
		if origin_x > end_x:
			temp = origin_x
			origin_x = end_x
			end_x = temp
		points = []
		points.append((origin_y, origin_x))
		low_xi = origin_x
		for j in range(1, num_seeds):
		
			seed_y, seed_x = int(origin_y + y_dist_between_points*j), np.random.randint(low=low_xi, high=end_x)
			new_im[seed_y, seed_x] = 1
			low_xi = seed_x
			points.append((seed_y, seed_x))

		points.append((end_y, end_x))

		point_pairs = list(pairwise(points))

		print(points)
		print(point_pairs)
		print('\n\n')

		for point in point_pairs:
			rr, cc = line(point[0][0], point[0][1], point[1][0], point[1][1])
			new_im[rr, cc] = 1

		plt.imsave(root_output_dir + 'LineSeed/' + str(i)+'.png', new_im, cmap=plt.get_cmap('gray'))
