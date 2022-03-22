# Third party imports
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
from skimage.morphology import disk, binary_erosion, erosion
from skimage.filters import median as md
from skimage.filters import rank

# Local imports
from utils import *

# Global setting for matplotlib
matplotlib.use('Agg')


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


def main_function(root_output_dir, struct_veins, ind):
	"""
	Orchestrator function for generating artificial hand vein-like structures on a black background.
	:param root_output_dir: Output directory to save results.
	:param i: I'th image in the list.
	:return: None
	"""

	for f_vein in struct_veins:
		thinned = binary_erosion(f_vein, disk(1))
		grey = make_greyscale(thinned)
		img = normalize_vein_intensity(grey)
		img = rank.mean(img, selem=disk(3))
		img = md(img, disk(3))
		img = rank.mean(img, selem=disk(2))
		img = md(img, disk(2))
		img = erosion(img, disk(2))
		img = rank.mean(img, selem=disk(3))
		img = md(img, disk(3))
		img = erosion(img, disk(1.5))
		save(root_output_dir, img, ind, struct_veins.index(f_vein), 'Grey')
