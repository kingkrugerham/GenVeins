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


# TODO: Add sparsity between seed veins!

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
		- Each pair of seed veins must be sufficiently separated (not in close proximity).
		- Each pair of seed veins must have a different angle (larger that 10 degrees different).
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


def verify_vein_spread(im):
	"""
	Verify that the random propagation algorithm introduced enough spread of the veins across the image.
	If not, image is discarded and a new one is created in its place.
	:param im: Image from which to calculate spread.
	:return: False if percentage of white pixels are within the given ranges.
	"""
	spread = np.count_nonzero(im)/2000.
	if spread < 0.35 or spread > 0.7:
		return False
	return True


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
	dilation_factor = np.random.choice([1, 1.5, 2], p=[0.3, 0.4, 0.3])
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
