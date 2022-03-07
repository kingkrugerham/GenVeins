# System imports
import os
import sys

# Third party imports
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
from skimage.io	import _io

# Local imports
from utils import *

# Global setting for matplotlib
matplotlib.use('Agg')

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
