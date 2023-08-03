# System imports
import sys

# Third party imports
import cv2
import numpy as np
from skimage import transform as tf

# Local imports
from utils import *
from test import add_random_objects

# Increase python default recursion limit to enable 
# search for acceptable hand vein image.
sys.setrecursionlimit(100000)


def generate_tree_seed_pairs(original_im):
	"""
	Generate a number of tree seed point pairs to connect with vein-like structures.
	The proposed algorithm is as follows:
		- Find vein and background coordinates in original_im.
		- Randomly choose number of tree seed pairs (either 0, 1 or 2) in favour of 1 tree vein.
		- Verify the vertical and horizontal distances between selected tree seed points are in acceptable range:
			(tree veins are typically much shorter than base/main veins).
	:param original_im: Image containing base veins from which to determine tree seed point pairs.
	:return: List of seed point pairs for tree veins.
	"""
	vein_coords, back_coords = find_coords(original_im)
	num_tree_seeds = np.random.choice([0, 1, 2], p=[0.3, 0.5, 0.2])
	tree_seed_pairs = []
	if not num_tree_seeds == 0:
		for _ in range(num_tree_seeds):
			seed_1 = vein_coords[np.random.randint(len(vein_coords))]
			seed_2 = back_coords[np.random.randint(len(back_coords))]
			accepted_range = np.arange(10, 20)
			while i_dist_between_seeds(seed_1, seed_2) not in accepted_range or \
				  j_dist_between_seeds(seed_1, seed_2) not in accepted_range:
				seed_1 = vein_coords[np.random.randint(len(vein_coords))]
				seed_2 = back_coords[np.random.randint(len(back_coords))]
			tree_seed_pairs.append(sorted([seed_1, seed_2]))
	return tree_seed_pairs


def generate_noise_seed_pairs(original_im):
	"""
	Generate a number of unconnected seed point pairs to connect with vein-like structures.
	The proposed algorithm is as follows:
		- Same selection criteria as tree veins (see docstring for generate_tree_seed_pairs(original_im)).
		- Unconnected veins are not required to be connected to any existing veins.
		- Unconnected veins are also defined a little shorter than branch veins, since the purpose of adding
			unconnected veins are to simulate noise obtained during acquisition (may or may not be veins).
	:param original_im: Image from which to determine unconnected seed point pairs.
	:return: List of seed point pairs for unconnected veins.
	"""
	_, back_coords = find_coords(original_im)
	num_objects = np.random.choice([0, 1, 2], p=[0.3, 0.5, 0.2])
	unconnected_seed_pairs = []
	if not num_objects == 0:
		for _ in range(num_objects):
			seed_1 = back_coords[np.random.randint(len(back_coords))]
			seed_2 = back_coords[np.random.randint(len(back_coords))]
			accepted_range = np.arange(10, 20)
			while i_dist_between_seeds(seed_1, seed_2) not in accepted_range or \
				  j_dist_between_seeds(seed_1, seed_2) not in accepted_range:
				seed_1 = back_coords[np.random.randint(len(back_coords))]
				seed_2 = back_coords[np.random.randint(len(back_coords))]
			unconnected_seed_pairs.append(sorted([seed_1, seed_2]))
	return unconnected_seed_pairs


def draw_tree_veins(original_im):
	"""
	Possibly add tree veins based on the current vein information. The proposed algorithm is as follows:
		- Tree veins propagate out of existing veins towards another random seed point not required to be 
			within the current vein structure.
		- They follow the same propagation algorithm as generating the seed veins.
		- Dilate tree veins by a random acceptable factor (typically thinner than main veins)
	:param original_im: Binary base veins image from which to determine tree seed pairs.
	:return: Black image with added tree veins (will be joined with base veins later in the algorithm).
	"""
	im = initialize_im()
	tree_point_pairs = generate_tree_seed_pairs(original_im)
	if len(tree_point_pairs) > 0:
		im = propagate_and_draw_veins(im, tree_point_pairs)
		im = dilate(im, 1.5)
	return im


def draw_unconnected_veins(original_im):
	"""
	Possibly add unconnected veins based on the current vein information. The proposed algorithm is as follows:
		- Unconnected veins are not required to be connected to the existing vein structure (simulate acquisition noise).
		- They follow the same propagation algorithm as generating the seed veins.
		- Dilate unconnected veins by an acceptable factor (typically thinner than base veins).
	:param original_im: Base vein image from which to determine unconnected seed point pairs.
	:return: Black image with added unconnected veins (will be joined to existing hand vein image later in the flow).
	"""
	im = initialize_im()
	unconnected_seed_pairs = generate_noise_seed_pairs(original_im)
	if len(unconnected_seed_pairs) > 0:
		im = propagate_and_draw_veins(im, unconnected_seed_pairs)
		im = dilate(im, 1.5)
	return im


def union_vein_ims(original_im, tree_veins, segmentation_noise):
	"""
	Union all the generated hand vein images (base veins, tree veins and unconnected veins).
	:param original_im: Seed vein image.
	:param tree_veins: Tree vein image.
	:param segmentation_noise: Unconnected vein image.
	:return: Final vein image.
	"""
	final_veins = initialize_im()
	for i in range(final_veins.shape[0]):
		for j in range(final_veins.shape[1]):
			if tree_veins[i, j] > 0 or segmentation_noise[i, j] > 0 or original_im[i, j] > 0:
				final_veins[i, j] = 1
	return final_veins
	

def main_function(root_output_dir, base_veins, ind):
	"""
	Orchestrator function for simulating multiple hand vein acquisitions from a given base vein image.
	Algorithm is defined as follows:
		- Draw tree/branch veins, where at least one seed point must coincide with a seed vein.
		- Randomly choose to add unconnected vein-like structures to the image (simulate segmentation noise).
		- Union the base/seed veins, branch veins and unconnected veins.
		- Repeat 4 times in order to acquire 4 simulated acquired hand vein images.
	:param root_output_dir: Root output directory of results.
	:param base_veins: Seed vein image.
	:param ind: Numbered individual (for naming output files).
	:return: List containing 4 simulated acquired hand vein images for individual # ind
	"""
	sims = []
	for _ in range(4):
		tree_veins = draw_tree_veins(base_veins)
		branch_veins = draw_unconnected_veins(base_veins)
		union_veins = union_vein_ims(base_veins, tree_veins, branch_veins)

		# Apply random rotation and translation
		angle = np.random.uniform(-1, 1)
		tx = np.random.uniform(-1, 1)
		ty = np.random.uniform(-1, 1)
		transformation_matrix = np.float32([[1,0,tx], [0,1,ty]])
		union_veins = cv2.warpAffine(union_veins, transformation_matrix, (union_veins.shape[1], union_veins.shape[0]), cv2.BORDER_WRAP)
		union_veins = tf.rotate(union_veins, angle, mode='wrap')

		sims.append(union_veins)
		# save(root_output_dir, union_veins, ind, num_sims, 'Final_Veins')  # uncomment if you want to see the generated base binary structures.
	return sims
