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


def generate_seed_pairs():
	"""
	Randomly add seed vein points (seed pairs). The proposed constraints for selecting seed pairs are as follows:
		- Seed pairs are located on different edges of the image.
		- Number of seed pairs will be either 2, 3 or 4 (resulting in 2, 3 or 4 seed veins)
		- Squared Euclidean Distance between seed points must be larger or equal to 30 (span enough of the image)
		- Vertical distance between seed points must be larger or equal to 40 (Ensure vertically oriented seed veins)
	:param im: Black image onto which to add the seed points.
	:param root_output_dir: Location to save images.
	:param i: ith image in the list.
	:return: Image containing seed points, seed point pairs.
	"""
	num_seed_points = np.random.choice([2, 3], p=[0.65, 0.35])
	image_border_ranges = [[(0, n) for n in range(0, 40)],
						   [(m, 39) for m in range(0, 50)],
						   [(49, n) for n in range(0, 40)], 
						   [(m, 0) for m in range(0, 50)]]
	seed_point_pairs = []
	for _ in range(num_seed_points):
		borders = np.random.choice(image_border_ranges, size=2, replace=False, p=[0.3, 0.2, 0.3, 0.2])
		seed_1 = borders[0][np.random.randint(len(borders[0]))]
		seed_2 = borders[1][np.random.randint(len(borders[1]))]
		while dist_between_seeds(seed_1, seed_2) < 35 or i_dist_between_seeds(seed_1, seed_2) < 35:
			seed_1 = borders[0][np.random.randint(len(borders[0]))]
			seed_2 = borders[1][np.random.randint(len(borders[1]))]
		if len(seed_point_pairs) > 0:
			while not verify_seed_vein(seed_point_pairs, seed_1, seed_2):
				borders = np.random.choice(image_border_ranges, size=2, replace=False, p=[0.3, 0.2, 0.3, 0.2])
				seed_1 = borders[0][np.random.randint(len(borders[0]))]
				seed_2 = borders[1][np.random.randint(len(borders[1]))]
				while dist_between_seeds(seed_1, seed_2) < 35 or i_dist_between_seeds(seed_1, seed_2) < 35:
					seed_1 = borders[0][np.random.randint(len(borders[0]))]
					seed_2 = borders[1][np.random.randint(len(borders[1]))]
		seed_point_pairs.append(sorted([seed_1, seed_2]))
	return seed_point_pairs


def generate_branch_seed_pairs(seed_veins):
	"""
	Generate a number of branch seed point pairs to connect with vein-like structures.
	:param im: Image from which to determine branch seed point pairs.
	:return: List of seed point pairs for branch veins.
	"""
	vein_coords = []
	for i in range(seed_veins.shape[0]):
		for j in range(seed_veins.shape[1]):
			if seed_veins[i, j] == 1:
				vein_coords.append((i, j))

	num_branch_seeds = np.random.choice([1, 2, 3], p=[0.4, 0.4, 0.2])
	branch_seed_pairs = []
	if not num_branch_seeds == 0:
		for _ in range(num_branch_seeds):
			seed_1 = vein_coords[np.random.randint(len(vein_coords))]
			seed_2 = vein_coords[np.random.randint(len(vein_coords))]
			while i_dist_between_seeds(seed_1, seed_2) < 10 or j_dist_between_seeds(seed_1, seed_2) < 5:
				seed_1 = vein_coords[np.random.randint(len(vein_coords))]
				seed_2 = vein_coords[np.random.randint(len(vein_coords))]
			branch_seed_pairs.append(sorted([seed_1, seed_2]))
	return branch_seed_pairs


def verify_seed_vein(seed_point_pairs, seed_1, seed_2):
	"""
	Check that seed pairs adhere to the following constraints:
		- Each pair of seed veins must be sufficiently separated (not in close proximity).
		- Each pair of seed veins must have a different angle (larger that 10 degrees different).
	"""
	angles = []
	for spp in seed_point_pairs:
		angles.append(angle_between_seeds(spp[0], spp[1]))
	new_angle = angle_between_seeds(seed_1, seed_2)
	angle_prod = list(product([new_angle], angles))
	accepted_angle_range = [v for v in range(10, 40)]
	if all([abs(v[0] - v[1]) in accepted_angle_range for v in angle_prod]):
		return True
	return False


def verify_vein_spread(im):
	"""
	Verify that the random propagation algorithm introduced enough spread of the veins across the image.
	If not, image is discarded and a new one is created in its place.
	:param im: Image from which to calculate spread.
	:return: False if percentage of white pixels are within the given ranges.
	"""
	vein_coords = []
	for i in range(im.shape[0]):
		for j in range(im.shape[1]):
			if im[i, j] == 1:
				vein_coords.append((i, j))
	spread = len(vein_coords)/2000.
	print(spread)
	if spread < 0.3 or spread > 0.7:
		return False
	return True


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


def draw_seed_veins(root_output_dir, ind):
	"""
	Function to connect seed point pairs with vein-like structures. The proposed algorithm is detailed in propagate_and_draw_veins():
	:param root_output_dir: Location to save images.
	:param i: ith image in the list.
	:return: Image containing seed veins, seed vein coordinates
	"""
	im = initialize_im()
	seed_veins_dir = root_output_dir + '1.SeedVeins/'
	make_output_dir(seed_veins_dir)
	seed_point_pairs = generate_seed_pairs()
	im = propagate_and_draw_veins(im, seed_point_pairs)
	dilation_factor = np.random.choice([1, 1.5, 2], p=[0.25, 0.3, 0.45])
	im = dilate(im, dilation_factor)
	plt.imsave(seed_veins_dir + 'person_'+str(ind)+'.png', im, cmap=plt.get_cmap('gray'))
	return im


def draw_branch_veins(root_output_dir, ind, seed_veins):
	"""
	Add branching veins based on the seed vein information. The proposed algorithm is as follows:
		- Branching veins run from one seed vein to another.
		- They follow the same propagation algorithm as generating the seed veins.
	:param root_output_dir: Location to save images.
	:param i: ith image in the list.
	:return: Image with added branch veins.
	"""
	im = initialize_im()
	branch_veins_dir = root_output_dir + '2.BranchVeins/'
	make_output_dir(branch_veins_dir)
	branch_point_pairs = generate_branch_seed_pairs(seed_veins)
	if len(branch_point_pairs) > 0:
		im = propagate_and_draw_veins(im, branch_point_pairs)
		dilation_factor = np.random.choice([1, 1.5])
		im = dilate(im, dilation_factor)
	plt.imsave(branch_veins_dir + 'person_'+str(ind)+'.png', im, cmap=plt.get_cmap('gray'))
	return im


def union_vein_ims(seed_veins, branch_veins):
	"""
	Union all the generated hand vein images.
	:param seed_veins: Seed vein image.
	:param branch_veins: Branch vein image.
	:return: Final vein image.
	"""
	final_veins = initialize_im()
	for i in range(final_veins.shape[0]):
		for j in range(final_veins.shape[1]):
			if seed_veins[i, j] > 0 or branch_veins[i, j] > 0:
				final_veins[i, j] = 1
	return final_veins


def main_function(root_output_dir, ind):
	"""
	Orchestrator function for generating artificial hand vein-like structures on a black background.
	:param root_output_dir: Output directory to save results.
	:param i: I'th image in the list.
	:return: None
	"""
	seed_veins = draw_seed_veins(root_output_dir, ind)
	branch_veins = draw_branch_veins(root_output_dir, ind, seed_veins)
	final_veins = union_vein_ims(seed_veins, branch_veins)
	while verify_vein_spread(final_veins) == False:
		seed_veins = draw_seed_veins(root_output_dir, ind)
		branch_veins = draw_branch_veins(root_output_dir, ind, seed_veins)
		final_veins = union_vein_ims(seed_veins, branch_veins)
	final_veins_dir = root_output_dir + '3.FinalVeins/'
	make_output_dir(final_veins_dir)
	plt.imsave(final_veins_dir + 'person_'+str(ind)+'.png', final_veins, cmap=plt.get_cmap('gray'))
	

if __name__ == '__main__':
	root_output_dir = 'C:/Users/User/Desktop/PhD_Files/Output/GANs/Lightning/'
	num_ims=100
	for ind in range(num_ims):
		main_function(root_output_dir, ind)
