# System imports
import sys
from itertools import product

# Third party imports
import matplotlib
from matplotlib import pyplot as plt
import numpy as np

# Local imports
from utils import *

# Global setting for matplotlib
matplotlib.use('Agg')

# Increase python default recursion limit to enable 
# search for acceptable hand vein image.
sys.setrecursionlimit(100000)


def generate_seed_pairs():
	"""
	Randomly add seed vein points (seed pairs). The proposed algorithm is as follows:
		- Seed pairs are located on different edges of the image. Favour horizontal, rather than vertical edges
			due to the typical direction of seed veins inside a hand.
		- Number of seed pairs will be either 2 or 3 (resulting in 2 or 3 seed veins - favour 2 seed veins).
		- Choose initial seed vein points and verify the following 2 conditions:
			~ Squared Euclidean Distance between seed points must be larger or equal to 35 (span enough of the image).
			~ Vertical distance between seed points must be larger or equal to 35 (Ensure vertically oriented seed veins).
		- Repeat for number of seed point pairs, while only adding another seed point pair if verify_seed_vein() is True.
			See method docstring for conditions of adding another acceptable seed point pair.
	:return: List of seed point pairs.
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
	The proposed algorithm is as follows:
		- Find seed vein coordinates.
		- Number of branch veins in [0, 1, 2, 3], favouring 1 and 2.
		- Branch veins are encouraged to flow in a vertical, rather than horizontal direction
	:param seed_veins: Image from which to determine branch seed point pairs.
	:return: List of seed point pairs for branch veins.
	"""
	vein_coords = []
	for i in range(seed_veins.shape[0]):
		for j in range(seed_veins.shape[1]):
			if seed_veins[i, j] == 1:
				vein_coords.append((i, j))

	num_branch_seeds = np.random.choice([0, 1, 2, 3], p=[0.2, 0.35, 0.3, 0.15])
	branch_seed_pairs = []
	if not num_branch_seeds == 0:
		for _ in range(num_branch_seeds):
			seed_1 = vein_coords[np.random.randint(len(vein_coords))]
			seed_2 = vein_coords[np.random.randint(len(vein_coords))]
			while i_dist_between_seeds(seed_1, seed_2) < 20:
				seed_1 = vein_coords[np.random.randint(len(vein_coords))]
				seed_2 = vein_coords[np.random.randint(len(vein_coords))]
			branch_seed_pairs.append(sorted([seed_1, seed_2]))
	return branch_seed_pairs


def verify_seed_vein(seed_point_pairs, seed_1, seed_2):
	"""
	Check that seed pairs adhere to the following constraints:
		- Each pair of seed veins must have a different angle than all existing seed veins (larger than 10 degrees).
			This is an attempt to mitigate the problem of adjacent seed veins 
			(as it is rarely the case with actual hand veins).
	:param seed_point_pairs: Exisiting seed point pairs from which to determine angle between new seed point pair.
	:param seed_1: New seed point 1.
	:param seed_2 New seed point 2.
	:return: True if angle between new seed point pair is different enough from all existing seed point pairs.
	"""
	angles = []
	for spp in seed_point_pairs:
		angles.append(angle_between_seeds(spp[0], spp[1]))
	new_angle = angle_between_seeds(seed_1, seed_2)
	angle_prod = list(product([new_angle], angles))
	accepted_angle_range = [v for v in range(10, 35)]
	if all([abs(v[0] - v[1]) in accepted_angle_range for v in angle_prod]):
		return True
	return False

# TODO: Add more sparsity between seed veins!
def verify_vein_spread(im):
	"""
	Verify that the random propagation algorithm introduced enough spread of the veins across the image.
	If not, image is discarded and a new one is created in its place until this function returns True.
	:param im: Image from which to calculate spread.
	:return: False if percentage of white pixels are within the given ranges.
	"""
	spread = np.count_nonzero(im)/2000.
	if spread < 0.35 or spread > 0.7:
		return False
	return True


def draw_seed_veins():
	"""
	Draw seed veins on a black image. The proposed algorithm is as follows:
		- Generate seed point pairs based on conditions documented in docstring of generate_seed_pairs().
		- Draw acceptable seed veins based on algorithm outlined in docstring of propagate_and_draw_veins().
		- Dilate seed veins by an acceptable factor (seed veins are typically thicker than branch and other veins).
	:return: Bianry image with added seed veins.
	"""
	im = initialize_im()
	seed_point_pairs = generate_seed_pairs()
	im = propagate_and_draw_veins(im, seed_point_pairs)
	im = dilate(im, 2)
	return im


def draw_branch_veins(seed_veins):
	"""
	Possibly draw branch veins on a black image based on the seed vein information. 
	The proposed algorithm is as follows:
		- Branch veins run from one seed vein to another.
		- They follow the same propagation algorithm as generating the seed veins.
		- Dilate branch veins by an acceptable factor (typically thinner than seed veins).
	:param seed_veins: Image containing seed veins from which to determine and draw branch veins.
	:return: Binary image with added branch veins.
	"""
	im = initialize_im()
	branch_point_pairs = generate_branch_seed_pairs(seed_veins)
	if len(branch_point_pairs) > 0:
		im = propagate_and_draw_veins(im, branch_point_pairs)
		im = dilate(im, 1.5)
	return im


def union_vein_ims(seed_veins, branch_veins):
	"""
	Union the generated seed and branch hand vein images.
	:param seed_veins: Seed vein image.
	:param branch_veins: Branch vein image.
	:return: Base vein image.
	"""
	base_veins = initialize_im()
	for i in range(base_veins.shape[0]):
		for j in range(base_veins.shape[1]):
			if seed_veins[i, j] > 0 or branch_veins[i, j] > 0:
				base_veins[i, j] = 1
	return base_veins


def main_function(root_output_dir, ind):
	"""
	Orchestrator function for generating base (seed + branch) veins on a black background. 
	The proposed algorithm is as follows (find detailed descriptions of each step in function docstrings):
		- Draw the first iteration of seed and branch veins on a black background.
		- Verify the obtained base veins are acceptable and continue to restart until this is so.
	:param root_output_dir: Output directory to save results.
	:param ind: Numbered individual in question.
	:return: Binary image containing base veins.
	"""
	seed_veins = draw_seed_veins()
	branch_veins = draw_branch_veins(seed_veins)
	base_veins = union_vein_ims(seed_veins, branch_veins)
	while verify_vein_spread(base_veins) == False:
		seed_veins = draw_seed_veins()
		branch_veins = draw_branch_veins(seed_veins)
		base_veins = union_vein_ims(seed_veins, branch_veins)
	save(root_output_dir, base_veins, ind, '', 'Base_Veins')
	return base_veins
