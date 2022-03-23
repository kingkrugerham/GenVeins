# Third party imports
import numpy as np
from skimage.morphology import disk, binary_erosion, erosion
from skimage.filters import median, rank

# Local imports
from utils import *


def make_greyscale(original_im):
	"""
	Function to facilitate the initial greyification of a binary hand vein image.
	The proposed algorithm is outlined as follows:
		- Initialize a black image of same shape as input.
		- Loop through original image and determine whether current pixel is a vein or background pixel.
		- Randomly choose greyscale vein or background intensity for current pixel from acceptable intensity ranges.
		- Add minor variation to choice.
		- Set corresponding pixel in new image equal to newly obtained intensity.
	:param original_im: Thinned version of a given binary hand vein image.
	:return: Initially greyified version of original_im.
	"""
	new_im = initialize_im()
	for i in range(original_im.shape[0]):
		for j in range(original_im.shape[1]):
			var = np.random.uniform(-0.03, 0.03)
			if original_im[i, j] > 0:
				vein_intensity = np.random.uniform(0.15, 0.4)
				new_im[i, j] = vein_intensity + var
			else:
				background_intensity = np.random.uniform(0.05, 0.14)
				new_im[i, j] = background_intensity + var
	return new_im


def normalize_vein_intensity(img):
	"""
	Function to normalize the vein intensity in order to avoid discontinuities introduced by initial greyification.
	The proposed algorithm is outlined as follows.
		- Find the average intensity of the hand vein pixels in the initially greyified image.
		- Set each vein pixel equal to the average intensity and add minor variation to avoid a monotone vein intensity.
	:param img: Initially greyified vein image (output of make_greyscale(original_im))
	:return: Normalized grey scale hand vein image.
	"""
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
	Orchestrator function for creating a grey-scale version of the simulated acquired hand vein-like images.
	The proposed algorithm is outlined as follows:
		- First apply minor erosion of the hand vein image, since subsequent smoothing will thicken vein structures.
		- Randomly greyify vein and background pixels by selecting from an acceptable intensity range.
		- Normalize the vein intensities to reduce discontinuity.
		- Apply successive smoothing with local mean and median filters.
		- Apply intermittent grey-scale erosion in order to avoid blob-like structures.
	:param root_output_dir: Output directory to save results.
	:param struct_veins: List of 4 simulated acquired hand veins for individual 'ind'.
	:param ind: Numbered individual.
	:return: None
	"""
	for i in range(len(struct_veins)):
		f_vein = struct_veins[i]
		img = binary_erosion(f_vein, disk(1))
		img = make_greyscale(img)
		img = normalize_vein_intensity(img)
		img = rank.mean(img, selem=disk(3))
		img = median(img, disk(3))
		img = rank.mean(img, selem=disk(2))
		img = median(img, disk(2))
		img = erosion(img, disk(2))
		img = rank.mean(img, selem=disk(3))
		img = median(img, disk(3))
		img = erosion(img, disk(1.5))
		save(root_output_dir, img, ind, i, 'Grey')
