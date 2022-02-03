# System imports
import os

# Third party imports
from mahotas import labeled as lb
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
from scipy.ndimage import measurements, filters, morphology
from scipy.linalg import basic as ln
from skimage.io	import _io
from skimage.exposure import equalize_adapthist
from skimage.filters import threshold_local, threshold_otsu, gaussian
from skimage import morphology
from skimage.transform import resize, _warps
from skimage.draw import line
from more_itertools import pairwise
from itertools import combinations

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


def create_intersection(root_input_dir, root_output_dir, num_ims):
	"""
	Generate binary image containing seed points to build connected hand vein-like structures from.
	The protocol is as follows:
		- Generate a black image of shape (50, 40)
		- Randomly select number of seed points between 5 and 10.
		- Randomly select a coordinate in the black image and set to 1. Repeat for number of seed points
	:param seed_output_loc: Location to save seed images.
	:param num_ims: Number of seed images to create.
	:return: None
	"""
	make_output_dir(root_output_dir + 'Intersect/')
	make_output_dir(root_output_dir + 'Dilate/')
	make_output_dir(root_output_dir + 'Max_Tree/')
	ims = os.listdir(root_input_dir)
	combs = list(combinations(ims, 2))
	np.random.shuffle(combs)
	rand_select = combs[:num_ims]
	for pair in rand_select:
		im1 = _io.imread(root_input_dir + pair[0], as_gray=True)
		im2 = _io.imread(root_input_dir + pair[1], as_gray=True)
		intersect = np.multiply(im1, im2)
		plt.imsave(root_output_dir + 'Intersect/' + pair[0] + '_' + pair[1] +'.png', intersect, cmap=plt.get_cmap('gray'))
		skel = morphology._skeletonize.skeletonize(intersect)
		plt.imsave(root_output_dir + 'Skel/' + pair[0] + '_' + pair[1] +'.png', skel, cmap=plt.get_cmap('gray'))
		max_tree_im = morphology.flood_fill(skel)
		# dil = dilation(skel, selem=selem.disk(3))
		plt.imsave(root_output_dir + 'Max_Tree/' + pair[0] + '_' + pair[1] +'.png', max_tree_im, cmap=plt.get_cmap('gray'))


if __name__ == '__main__':
	root_input_dir = 'C:/Users/User/Desktop/PhD_Files/Input/Bosphorus/Binary/'
	root_output_loc = 'C:/Users/User/Desktop/PhD_Files/Output/GANs/Intersect/'
	create_intersection(root_input_dir, root_output_loc, num_ims=100)
	# create_skeleton(root_output_loc + 'Seed/', root_output_loc + 'Skel/')