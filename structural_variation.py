# System imports
from multiprocessing.connection import wait
import os
import sys
import time
from itertools import product

# Third party imports
from mahotas import border, labeled as lb
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
from scipy.ndimage import measurements, filters, morphology
from scipy.linalg import basic as ln
from skimage.io	import _io
from skimage.filters import threshold_local, threshold_otsu, gaussian
from skimage.morphology import convex_hull_image, selem, grey, _skeletonize, reconstruction, binary_dilation
from skimage.transform import resize, _warps
from skimage.draw import line
from more_itertools import pairwise

# Global setting for matplotlib
matplotlib.use('Agg')
sys.setrecursionlimit(100000)


def make_output_dir(loc):
	"""
	Create output directory if doesn't exist.
	:param loc: Output directory to check or create.
	:return: None
	"""
	if not os.path.exists(loc):
		os.makedirs(loc)


def dist_between_seeds(seed_1, seed_2):
	"""
	Wrapper function for Squared Euclidean Distance between seed points.
	:param seed_1: First seed point.
	:param seed_2: Second seed point.
	:return: Squared Euclidean Distance between seed_1 and seed_2
	"""
	return np.sqrt((seed_1[0] - seed_2[0])**2 + (seed_1[1] - seed_2[1])**2)


def i_dist_between_seeds(seed_1, seed_2):
	"""
	Calculate the vertical distance between seed points.
	:param seed_1: First seed point.
	:param seed_2: Second seed point.
	:return: Vertical distance between seed_1 and seed_2.
	"""
	return abs(seed_1[0] - seed_2[0])


def j_dist_between_seeds(seed_1, seed_2):
	"""
	Calculate the horizontal distance between seed points.
	:param seed_1: First seed point.
	:param seed_2: Second seed point.
	:return: Horizontal distance between seed_1 and seed_2.
	"""
	return abs(seed_1[1] - seed_2[1])


def propagate_and_draw_veins(im, seed_point_pairs):
	"""
	Algorithm for drawing veins between seed points.
		- Randomly choose number of connection points in line.
		- Calculate horizontal and vertical distance between seed points.
		- The vertical distance between connection points are constant (based on distance between and number of seed points).
		- The horizontal path followed is based on the distance between and number of seed points, with random variation for each step.
		- The connection points are constrained to not overflow out of the image and only run between seed points.
	:param im: Seed point image onto which to draw the vein-like structures.
	:param i: ith image in the list.
	:param seed_point_pairs: Seed pairs which to connect with vein-like structures.
	:return: Image containing veins, vein coordinates.
	"""
	for seed_point_pair in sorted(seed_point_pairs):
		num_connection_points = np.random.randint(low=5, high=15)
		seed_1_i, seed_1_j = seed_point_pair[0][0], seed_point_pair[0][1]
		seed_2_i, seed_2_j = seed_point_pair[1][0], seed_point_pair[1][1]
		i_dist_between_points = (seed_2_i - seed_1_i)/num_connection_points
		j_dist_between_points = (seed_2_j - seed_1_j)/num_connection_points
		points = [(seed_point_pair[0])]
		for m in range(1, num_connection_points):
			j_dist_random_addition_factor = np.random.choice([-2, -1, 1, 2])
			if np.random.random() > 0.3:
				seed_m_j = int(seed_1_j + j_dist_between_points*m) + j_dist_random_addition_factor
			else:
				seed_m_j = int(seed_1_j + j_dist_between_points*m)
			seed_m_i = int(seed_1_i + i_dist_between_points*m)	
			if seed_m_i > 49: seed_m_i = 49
			elif seed_m_i <= 0: seed_m_i = 1
			if seed_m_j > 39: seed_m_j = 39
			elif seed_m_j <= 0: seed_m_j = 1
			points.append((seed_m_i, seed_m_j))
		points.append(seed_point_pair[1])
		point_pairs = list(pairwise(points))
		for point in point_pairs:
			rr, cc = line(point[0][0], point[0][1], point[1][0], point[1][1])
			im[rr, cc] = 1
	return im


def dilate(im, size):
	"""
	Wrapper function to dilate given image with a certain size.
	:param root_output_dir: Location to save images.
	:param im: Image to dilate.
	:param size: Dilation size.
	:return: Dilated image.
	"""
	return binary_dilation(im, selem=selem.disk(size))


def initialize_im():
	"""
	Initialize a black image which will be populated with hand vein-like structures.
	:return: Black image of shape (50, 40).
	"""
	return np.zeros((50, 40))


def generate_tree_seed_pairs(original_im):
	"""
	Generate a number of tree seed point pairs to connect with vein-like structures.
	:param im: Image from which to determine tree seed point pairs.
	:return: List of seed point pairs for tree veins.
	"""
	vein_coords, back_coords = [], []
	for i in range(original_im.shape[0]):
		for j in range(original_im.shape[1]):
			if original_im[i, j] == 1:
				vein_coords.append((i, j))
			else:
				back_coords.append((i, j))

	num_tree_seeds = np.random.choice([0, 1, 2], p=[0.25, 0.45, 0.3])
	tree_seed_pairs = []
	if not num_tree_seeds == 0:
		for _ in range(num_tree_seeds):
			seed_1 = vein_coords[np.random.randint(len(vein_coords))]
			seed_2 = back_coords[np.random.randint(len(back_coords))]
			accepted_range = np.arange(5, 15)
			while i_dist_between_seeds(seed_1, seed_2) not in accepted_range or \
				  j_dist_between_seeds(seed_1, seed_2) not in accepted_range:
				seed_1 = vein_coords[np.random.randint(len(vein_coords))]
				seed_2 = back_coords[np.random.randint(len(back_coords))]
			tree_seed_pairs.append(sorted([seed_1, seed_2]))
	return tree_seed_pairs


def generate_noise_seed_pairs(original_im):
	"""
	Generate a number of unconnected seed point pairs to connect with vein-like structures.
	:param im: Image from which to determine unconnected seed point pairs.
	:return: List of seed point pairs for unconnected veins.
	"""
	vein_coords, back_coords = [], []
	for i in range(original_im.shape[0]):
		for j in range(original_im.shape[1]):
			if original_im[i, j] == 1:
				vein_coords.append((i, j))
			else:
				back_coords.append((i, j))

	num_objects = np.random.choice([0, 1, 2], p=[0.25, 0.45, 0.3])
	unconnected_seed_pairs = []
	if not num_objects == 0:
		for _ in range(num_objects):
			seed_1 = back_coords[np.random.randint(len(back_coords))]
			seed_2 = back_coords[np.random.randint(len(back_coords))]
			while i_dist_between_seeds(seed_1, seed_2) > 10 or j_dist_between_seeds(seed_1, seed_2) > 5:
				seed_1 = back_coords[np.random.randint(len(back_coords))]
				seed_2 = back_coords[np.random.randint(len(back_coords))]
			unconnected_seed_pairs.append(sorted([seed_1, seed_2]))
	return unconnected_seed_pairs


def draw_tree_veins(original_im):
	"""
	Add tree veins based on the current vein information. The proposed algorithm is as follows:
		- Tree veins propagate out of existing veins towards another random seed point not within the current vein structure.
		- They follow the same propagation algorithm as generating the seed veins.
	:param root_output_dir: Location to save images.
	:param i: ith image in the list.
	:return: Image with added tree veins.
	"""
	im = initialize_im()
	tree_point_pairs = generate_tree_seed_pairs(original_im)
	if len(tree_point_pairs) > 0:
		im = propagate_and_draw_veins(im, tree_point_pairs)
		dilation_factor = np.random.choice([1, 1.5], p=[0.55, 0.45])
		im = dilate(im, dilation_factor)
	return im


def simulate_segmentation_noise(original_im):
	"""
	Add unconnected veins based on the current vein information. The proposed algorithm is as follows:
		- Unconnected veins are not connected to the existing vein structure (introduces some noise into the image).
		- They follow the same propagation algorithm as generating the seed veins.
	:param root_output_dir: Location to save images.
	:param i: ith image in the list.
	:return: Image with added tree veins.
	"""
	im = initialize_im()
	unconnected_seed_pairs = generate_noise_seed_pairs(original_im)
	if len(unconnected_seed_pairs) > 0:
		im = propagate_and_draw_veins(im, unconnected_seed_pairs)
		dilation_factor = np.random.choice([1, 1.5], p=[0.65, 0.35])
		im = dilate(im, dilation_factor)
	return im


def union_vein_ims(original_im, tree_veins, segmentation_noise):
	"""
	Union all the generated hand vein images.
	:param seed_veins: Seed vein image.
	:param branch_veins: Branch vein image.
	:return: Final vein image.
	"""
	final_veins = initialize_im()
	for i in range(final_veins.shape[0]):
		for j in range(final_veins.shape[1]):
			if tree_veins[i, j] > 0 or segmentation_noise[i, j] > 0 or original_im[i, j] > 0:
				final_veins[i, j] = 1
	return final_veins


def main_function(root_input_dir, root_output_dir, im):
	"""
	Orchestrator function for generating artificial hand vein-like structures on a black background.
	:param root_output_dir: Output directory to save results.
	:param i: I'th image in the list.
	:return: None
	"""
	for num_sims in range(4):
		original_im = _io.imread(root_input_dir + im, as_gray=True)
		tree_veins = draw_tree_veins(original_im)
		branch_veins = simulate_segmentation_noise(original_im)
		final_veins = union_vein_ims(original_im, tree_veins, branch_veins)
		final_veins_dir = root_output_dir + '4.SimulatedAcquisition/'
		make_output_dir(final_veins_dir)
		plt.imsave(final_veins_dir + im.replace('.png', '') + '_{}'.format(str(num_sims)) + '.png', final_veins, cmap=plt.get_cmap('gray'))
	

if __name__ == '__main__':
	root_input_dir = 'C:/Users/User/Desktop/PhD_Files/Output/GANs/Lightning/3.FinalVeins/'
	root_output_dir = 'C:/Users/User/Desktop/PhD_Files/Output/GANs/Lightning/'
	ims = os.listdir(root_input_dir)
	for im in ims:
		main_function(root_input_dir, root_output_dir, im)
