# System imports
from operator import add
import os
import sys
from itertools import combinations, product

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


def propagate_and_draw_veins(im, i, seed_point_pairs):
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


def generate_branch_seed_pairs(im):
	"""
	Generate a number of branch seed point pairs to connect with vein-like structures.
	:param im: Image from which to determine branch seed point pairs.
	:return: List of seed point pairs for branch veins.
	"""
	vein_coords = []
	for i in range(im.shape[0]):
		for j in range(im.shape[1]):
			if im[i, j] == 1:
				vein_coords.append((i, j))

	num_branch_seeds = np.random.choice([1, 2, 3])
	branch_seed_pairs = []
	for _ in range(num_branch_seeds):
		seed_1 = vein_coords[np.random.randint(len(vein_coords))]
		seed_2 = vein_coords[np.random.randint(len(vein_coords))]
		while i_dist_between_seeds(seed_1, seed_2) < 10:
			seed_1 = vein_coords[np.random.randint(len(vein_coords))]
			seed_2 = vein_coords[np.random.randint(len(vein_coords))]
		branch_seed_pairs.append(sorted([seed_1, seed_2]))
	return branch_seed_pairs


def generate_tree_seed_pairs(im):
	"""
	Generate a number of tree seed point pairs to connect with vein-like structures.
	:param im: Image from which to determine tree seed point pairs.
	:return: List of seed point pairs for tree veins.
	"""
	vein_coords, back_coords = [], []
	for i in range(im.shape[0]):
		for j in range(im.shape[1]):
			if im[i, j] == 1:
				vein_coords.append((i, j))
			else:
				back_coords.append((i, j))

	num_tree_seeds = np.random.choice([1, 2, 3])
	tree_seed_pairs = []
	for _ in range(num_tree_seeds):
		seed_1 = vein_coords[np.random.randint(len(vein_coords))]
		seed_2 = back_coords[np.random.randint(len(back_coords))]
		while i_dist_between_seeds(seed_1, seed_2) < 10 or j_dist_between_seeds(seed_1, seed_2) < 10:
			seed_1 = vein_coords[np.random.randint(len(vein_coords))]
			seed_2 = back_coords[np.random.randint(len(back_coords))]
		tree_seed_pairs.append(sorted([seed_1, seed_2]))
	return tree_seed_pairs


def generate_unconnected_seed_pairs(im):
	"""
	Generate a number of unconnected seed point pairs to connect with vein-like structures.
	:param im: Image from which to determine unconnected seed point pairs.
	:return: List of seed point pairs for unconnected veins.
	"""
	vein_coords, back_coords = [], []
	for i in range(im.shape[0]):
		for j in range(im.shape[1]):
			if im[i, j] == 1:
				vein_coords.append((i, j))
			else:
				back_coords.append((i, j))

	num_unconnected_seeds = np.random.choice([1, 2, 3])
	unconnected_seed_pairs = []
	for _ in range(num_unconnected_seeds):
		seed_1 = back_coords[np.random.randint(len(back_coords))]
		seed_2 = back_coords[np.random.randint(len(back_coords))]
		while i_dist_between_seeds(seed_1, seed_2) > 15 or j_dist_between_seeds(seed_1, seed_2) > 15:
			seed_1 = back_coords[np.random.randint(len(back_coords))]
			seed_2 = back_coords[np.random.randint(len(back_coords))]
		unconnected_seed_pairs.append(sorted([seed_1, seed_2]))
	return unconnected_seed_pairs


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


def initialize_im():
	"""
	Initialize a black image which will be populated with hand vein-like structures.
	:return: Black image of shape (50, 40).
	"""
	return np.zeros((50, 40))


def generate_seed_pairs(im, root_output_dir, i):
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
	seed_points_dir = root_output_dir + '1.SeedVeinPoints/'
	make_output_dir(seed_points_dir)
	num_seed_points = np.random.choice([1, 2, 3])
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
		im[seed_1] = 1
		im[seed_2] = 1
	plt.imsave(seed_points_dir + 'person_'+str(i)+'.png', im, cmap=plt.get_cmap('gray'))
	return im, seed_point_pairs


def draw_seed_veins(im, root_output_dir, i):
	"""
	Function to connect seed point pairs with vein-like structures. The proposed algorithm is detailed in propagate_and_draw_veins():
	:param im: Seed point image onto which to draw the vein-like structures.
	:param root_output_dir: Location to save images.
	:param i: ith image in the list.
	:return: Image containing seed veins, seed vein coordinates
	"""
	seed_veins_dir = root_output_dir + '2.SeedVeins/'
	make_output_dir(seed_veins_dir)
	im, seed_point_pairs = generate_seed_pairs(im, root_output_dir, i)
	im = propagate_and_draw_veins(im, i, seed_point_pairs)
	plt.imsave(seed_veins_dir + 'person_'+str(i)+'.png', im, cmap=plt.get_cmap('gray'))
	return im


def draw_branch_veins(im, root_output_dir, i):
	"""
	Add branching veins based on the seed vein information. The proposed algorithm is as follows:
		- Branching veins run from one seed vein to another.
		- They follow the same propagation algorithm as generating the seed veins.
	:param im: Seed vein image onto which to draw the branch vein-like structures.
	:param root_output_dir: Location to save images.
	:param i: ith image in the list.
	:return: Image with added branch veins.
	"""
	branch_veins_dir = root_output_dir + '3.AddedBranchVeins/'
	make_output_dir(branch_veins_dir)
	branch_point_pairs = generate_branch_seed_pairs(im)
	im = propagate_and_draw_veins(im, i, branch_point_pairs)
	plt.imsave(branch_veins_dir + 'person_'+str(i)+'.png', im, cmap=plt.get_cmap('gray'))
	return im


def draw_tree_veins(im, root_output_dir, i):
	"""
	Add tree veins based on the current vein information. The proposed algorithm is as follows:
		- Tree veins propagate out of existing veins towards another random seed point not within the current vein structure.
		- They follow the same propagation algorithm as generating the seed veins.
	:param im: Image onto which to draw the tree vein-like structures.
	:param root_output_dir: Location to save images.
	:param i: ith image in the list.
	:return: Image with added tree veins.
	"""
	tree_veins_dir = root_output_dir + '4.AddedTreeVeins/'
	make_output_dir(tree_veins_dir)
	tree_point_pairs = generate_tree_seed_pairs(im)
	im = propagate_and_draw_veins(im, i, tree_point_pairs)
	plt.imsave(tree_veins_dir + 'person_'+str(i)+'.png', im, cmap=plt.get_cmap('gray'))
	return im


def draw_unconnected_veins(im, root_output_dir, i):
	"""
	Add unconnected veins based on the current vein information. The proposed algorithm is as follows:
		- Unconnected veins are not connected to the existing vein structure (introduces some noise into the image).
		- They follow the same propagation algorithm as generating the seed veins.
	:param im: Image onto which to draw the unconnected vein-like structures.
	:param root_output_dir: Location to save images.
	:param i: ith image in the list.
	:return: Image with added tree veins.
	"""
	unconnected_veins_dir = root_output_dir + '5.AddedUnconnectedVeins/'
	make_output_dir(unconnected_veins_dir)
	unconnected_seed_pairs = generate_unconnected_seed_pairs(im)
	im = propagate_and_draw_veins(im, i, unconnected_seed_pairs)
	plt.imsave(unconnected_veins_dir + 'person_'+str(i)+'.png', im, cmap=plt.get_cmap('gray'))
	return im


def dilate(im, root_output_dir, i):
	dilated_seed_veins_dir = root_output_dir + '6.DilatedVeins/'
	make_output_dir(dilated_seed_veins_dir)
	im = binary_dilation(im, selem=selem.disk(2))
	plt.imsave(dilated_seed_veins_dir + 'person_'+str(i)+'.png', im, cmap=plt.get_cmap('gray'))
	return im


def main(root_output_dir):
	"""
	Orchestrator function for generating artificial hand vein-like structures on a black background.
	:return: None
	"""
	for i in range(num_ims):
		im = initialize_im()
		im = draw_seed_veins(im, root_output_dir, i)
		im = draw_branch_veins(im, root_output_dir, i)
		im = draw_tree_veins(im, root_output_dir, i)
		im = dilate(im, root_output_dir, i)

		
if __name__ == '__main__':
	root_output_dir = 'C:/Users/User/Desktop/PhD_Files/Output/GANs/Lightning/'
	num_ims=100
	main(root_output_dir)


# choices = ['exp', 'inv_exp']
# 			distr_1 = np.random.choice(choices)
# 			if distr_1 == 'exp':
# 				vals_1 = [v for v in range(1, len(borders[0]))]
# 				probs_1 = [1./n for n in range(1, len(borders[0]))]
# 				probs_1_norm = [float(p)/sum(probs_1) for p in probs_1]
# 				seed_1 = borders[0][np.random.choice(vals_1, p=probs_1_norm)]
# 				vals_2 = [v for v in range(1, len(borders[1]))]
# 				probs_2 = [1./n for n in range(len(borders[1]) - 1, 0, -1)]
# 				probs_2_norm = [float(p)/sum(probs_2) for p in probs_2]
# 				seed_2 = borders[1][np.random.choice(vals_2, p=probs_2_norm)]
# 			else:
# 				vals_1 = [v for v in range(1, len(borders[1]))]
# 				probs_1= [1./n for n in range(1, len(borders[1]))]
# 				probs_1_norm = [float(p)/sum(probs_1) for p in probs_1]
# 				seed_1 = borders[1][np.random.choice(vals_1, p=probs_1_norm)]
# 				vals_2 = [v for v in range(1, len(borders[0]))]
# 				probs_2 = [1./n for n in range(len(borders[0])-1, 0, -1)]
# 				probs_2_norm = [float(p)/sum(probs_2) for p in probs_2]
# 				seed_2 = borders[0][np.random.choice(vals_2, p=probs_2_norm)]