# System imports
import sys

# Third party imports
import numpy as np

# Local imports
from utils import *

# Increase python default recursion limit to enable 
# search for acceptable hand vein image.
sys.setrecursionlimit(100000)


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
	for num_sims in range(4):
		tree_veins = draw_tree_veins(base_veins)
		branch_veins = simulate_segmentation_noise(base_veins)
		final_veins = union_vein_ims(base_veins, tree_veins, branch_veins)
		sims.append(final_veins)
		save(root_output_dir, final_veins, ind, num_sims, 'Final_Veins')
	return sims
