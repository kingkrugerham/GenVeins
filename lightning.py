# System imports
import os

# Third party imports
from mahotas import border, labeled as lb
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
from scipy.ndimage import measurements, filters, morphology
from scipy.linalg import basic as ln
from skimage.io	import _io
from skimage.exposure import equalize_adapthist
from skimage.filters import threshold_local, threshold_otsu, gaussian
from skimage.morphology import convex_hull_image, selem, grey, _skeletonize, reconstruction, binary_dilation
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


def initialize_im():
	"""
	Initialize a black image which will be populated with hand vein-like structures.
	:return: Black image of shape (50, 40).
	"""
	return np.zeros((50, 40))


def generate_seed_vein_points(im, root_output_dir, i):
	"""
	Randomly add seed vein points. The proposed algorithm is as follows:
		- Seed vein points are located on the edge of the image.
		- Number of seed vein points will be either 4, 6 or 8.
		- Seed vein points on the same edge will not be paired.
	:param im: Black image onto which to add the seed points.
	:param root_output_dir: Location to save images.
	:param i: ith image in the list.
	:return: Image containing seed points, seed point pairs.
	"""
	seed_points_dir = root_output_dir + '1.SeedVeinPoints/'
	make_output_dir(seed_points_dir)
	num_seed_points = np.random.choice([3, 4, 5])
	image_border_ranges = [[(0, n) for n in range(0, 40)],
						   [(m, 39) for m in range(0, 50)],
						   [(49, n) for n in range(0, 40)],
						   [(m, 0) for m in range(0, 50)]]
	seed_point_pairs = []
	for _ in range(num_seed_points):
		borders = np.random.choice(image_border_ranges, size=2, replace=False)
		seed_1 = borders[0][np.random.randint(len(borders[0]))]
		seed_2 = borders[1][np.random.randint(len(borders[1]))]
		while np.sqrt((seed_1[0] - seed_2[0])**2 + (seed_1[1] - seed_2[1])**2) < 30:
			seed_1 = borders[0][np.random.randint(len(borders[0]))]
			seed_2 = borders[1][np.random.randint(len(borders[1]))]
		seed_point_pairs.append(sorted([seed_1, seed_2]))
		im[seed_1] = 1
		im[seed_2] = 1
	plt.imsave(seed_points_dir + 'person_'+str(i)+'.png', im, cmap=plt.get_cmap('gray'))
	return im, seed_point_pairs


def draw_seed_veins(im, root_output_dir, i, seed_point_pairs):
	"""
	Function to connect seed point pairs with vein-like structures. The proposed algorithm is as follows:
		- 
	:param im: Seed point image onto which to draw the vein-like structures.
	:param root_output_dir: Location to save images.
	:param i: ith image in the list.
	:param seed_point_pairs: Seed pairs which to connect with vein-like structures.
	:return: Image containing seed veins.
	"""
	seed_veins_dir = root_output_dir + '2.SeedVeins/'
	make_output_dir(seed_veins_dir)
	for seed_point_pair in sorted(seed_point_pairs):
		num_connection_points = np.random.randint(low=5, high=15)
		seed_1_i, seed_1_j = seed_point_pair[0][0], seed_point_pair[0][1]
		seed_2_i, seed_2_j = seed_point_pair[1][0], seed_point_pair[1][1]
		i_dist_between_points = (seed_2_i - seed_1_i)/num_connection_points
		j_dist_between_points = (seed_2_j - seed_1_j)/num_connection_points
		points = [(seed_point_pair[0])]
		for m in range(1, num_connection_points):
			seed_m_i = int(seed_1_i + i_dist_between_points*m)
			seed_m_j = int(seed_1_j + j_dist_between_points*m) + np.random.randint(low=-3, high=3)
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
	plt.imsave(seed_veins_dir + 'person_'+str(i)+'.png', im, cmap=plt.get_cmap('gray'))
	return im


def dilate(im, root_output_dir, i):
	dilated_seed_veins_dir = root_output_dir + '3.DilatedSeedVeins/'
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
		im, seed_point_pairs = generate_seed_vein_points(im, root_output_dir, i)
		im = draw_seed_veins(im, root_output_dir, i, seed_point_pairs)
		im = dilate(im, root_output_dir, i)

		
if __name__ == '__main__':
	root_output_dir = 'C:/Users/User/Desktop/PhD_Files/Output/GANs/Lightning/'
	num_ims=100
	main(root_output_dir)
