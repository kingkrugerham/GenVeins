# System imports
import os

# Third party imports
import numpy as np
from scipy.linalg import basic as ln
from skimage.morphology import selem, binary_dilation
from skimage.draw import line
from more_itertools import pairwise


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


def angle_between_seeds(seed_1, seed_2):
	"""
	Calculate absolute angle between seed points.
	:param seed_1: First seed point.
	:param seed_2: Second seed point.
	:return: Absolute angle in degrees between seed_1 and seed_2.
	"""
	A = [[1, seed_1[0]], [1, seed_2[0]]]
	b = [seed_1[1], seed_2[1]]
	sol, _, _, _ = ln.lstsq(A,b)
	return np.round(np.degrees(np.arctan(sol[1])))


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
			j_dist_random_addition_factor = np.random.choice([-2, -1, 1, 2], p=[0.3, 0.2, 0.2, 0.3])
			if np.random.random() > 0.25:
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
	