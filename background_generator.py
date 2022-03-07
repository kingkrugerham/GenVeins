# System imports
import os

# Third party imports
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
from skimage.io	import _io
from skimage.morphology import disk, binary_erosion, erosion
from skimage.filters import median as md
from skimage.filters import rank

# Local imports
from utils import *

# Global setting for matplotlib
matplotlib.use('Agg')


def erode(original_im):
	return binary_erosion(original_im, disk(1))

def make_greyscale(original_im):
	new_im = initialize_im()
	for i in range(original_im.shape[0]):
		for j in range(original_im.shape[1]):
			vein_intensity = np.random.uniform(0.15, 0.4)
			background_intensity = np.random.uniform(0.05, 0.14)
			var = np.random.uniform(-0.03, 0.03)
			if original_im[i, j] > 0:
				new_im[i, j] = vein_intensity + var
			else:
				new_im[i, j] = background_intensity + var
	return new_im

def normalize_vein_intensity(img):
	new_im = initialize_im()
	avg_vein_intensity = np.mean([img[x, y] for x in range(img.shape[0]) for y in range(img.shape[1]) if img[x, y] >= 0.18])
	for i in range(img.shape[0]):
		for j in range(img.shape[1]):
			var = np.random.uniform(-0.03, 0.03)
			if img[i, j] >= 0.18:
				new_im[i, j] = avg_vein_intensity + var
			else:
				new_im[i, j] = img[i, j]
	return new_im

def grey_thin(img):
	return erosion(img, disk(2))


def smoothen(grey_veins):
	return md(grey_veins, disk(3))


def main_function(root_input_dir, root_output_dir, im):
	"""
	Orchestrator function for generating artificial hand vein-like structures on a black background.
	:param root_output_dir: Output directory to save results.
	:param i: I'th image in the list.
	:return: None
	"""
	original_im = _io.imread(root_input_dir + im, as_gray=True)
	thinned = erode(original_im)
	grey = make_greyscale(thinned)
	img = normalize_vein_intensity(grey)
	
	img = rank.mean(img, selem=disk(3))
	img = md(img, disk(3))
	img = rank.mean(img, selem=disk(2))
	img = md(img, disk(2))
	img = grey_thin(img)
	img = rank.mean(img, selem=disk(3))
	img = md(img, disk(3))
	img = erosion(img, disk(1.5))
	
	

	grey_dir = root_output_dir + '5.Greyscale/'
	make_output_dir(grey_dir)
	plt.imsave(grey_dir + im, img, cmap=plt.get_cmap('gray'))


if __name__ == '__main__':
	root_input_dir = 'C:/Users/User/Desktop/PhD_Files/Output/GANs/Lightning/4.SimulatedAcquisition/'
	root_output_dir = 'C:/Users/User/Desktop/PhD_Files/Output/GANs/Lightning/'
	ims = os.listdir(root_input_dir)
	for im in ims:
		main_function(root_input_dir, root_output_dir, im)