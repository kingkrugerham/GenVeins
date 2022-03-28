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
from skimage.morphology import convex_hull_image, selem, grey, _skeletonize, reconstruction, binary
from skimage.transform import resize, _warps

# Global setting for matplotlib
matplotlib.use('Agg')


class VeinsNetDataPreprocessor:
	"""
	A class used to preprocess the input data for the neural networks in veinsnet.py:
		- Find the ROI and save.
		- Apply CLAHE to ROI and save.
		- Binarise ROI and save.

	
	Attributes
	----------
	net: Employed neural network architecture.
	base_dir: Base directory of the system.
	database: Hand vein database in question.
	input_location: Input directory of images to preprocess.


	Methods
	-------
	set_input_location()
		Set the location of the original images of the given database.

	is_preprocessing_complete()
		Checks whether all 3 steps of preprocessing are completed for the given database.

	read_input_image(input_location)
		Read input image of the given database in grayscale format.

	save_image(name, prep, image)
		Save preprocessed image in correct location. If location does not exist yet, it is created.

	apply_gaussian_smoothing(image, factor)
		Apply gaussian smoothing to the input image.

	threshold(image)
		Apply Otsu's threshold to the input image.

	convex_hull_image(image)
		Find the convex hull of the object in the binary input image.

	center_of_mass(image)
		Find the center of mass of the object in the binary input image.

	extract_roi_wilches(original_im, convex_hull_im, center_of_mass)
		The region of interest is defined as the largest rectangle that fits 
		into the convex hull. The approximate aspect ratio of the ROI is 0.8
		in order to correspond to the aspect ratio of the Bosphorus ROI,
		and is centered on the center of mass of the convex hull. 
		Only used within the context of the Wilches database.

	apply_clahe(self, image, size, limit)
		Apply CLAHE to the input image.

	adaptive_threshold(image)
		Adaptive (local) thresholding to find the mask for morphological reconstruction.

	laplacian(image)
		Laplacian has strong responses near intensity slopes, steps and isolated pixels. 
		This is the first seed image for morphological reconstruction.

	black_top_hat(image)
		The bottom hat transformation detects narrow dark structures, which is ideal for identifying vein pixels.
		This is the second image for morphological reconstruction.

	intersection(seed1, seed2)
		Find the morphological skeleton of the intersection of the 2 obtained seed images.

	reconstruct(seed, mask)
		Find the morphological reconstruction of the final seed image, constrained by the mask image. First,
		we make sure the seed image contains no white pixels where the mask does not in order to avoid reconstruction
		errors. Secondly, there are a small number of cases where the seed and mask image do not overlap at all. In these
		cases, we are forced to return the mask image as the final binarised version.

	resize_for_network(image)
		Resizes the final binarized hand vein images to the expected size by the neural networks. 
		Only applicable within the context of the Wilches database, since for the Bosphorus database
		the final images are already correct size.

	erosion(image)
		Wrapper function for binary erosion. Only used within the context of the Bosphorus database.

	unify_background(threshold_im, convex_hull_im)
		Due to the image acquisition protocol, there is a circle around the hand
		after the application of Otsu's thresholding, which must be unified with the
		actual background of the hand. Only used within the context of the Bosphorus database.

	clean(unified_image)
		Remove leftover objects from the background and hand. Pad the 
		unified image with background values in order to ensure the hand
		is fully contained in the image. Only used within the context of the Bosphorus database.

	remove_fingers(cleaned_image)
		Remove the four fingers with binary opening. Only used within the context of the Bosphorus database.

	find_hand_boundary(cleaned_image)
		Find the hand boundary. This will be used to identify the intersection
		points between the fingers, which will subsequenctly be used to calculate the
		orientation of the hand. Only used within the context of the Bosphorus database.

	overlay(fingers_removed, hand_boundary)
		Find the intersection between the binary hand with fingers removed and the hand boundary.
		Only used within the context of the Bosphorus database.

	find_finger_valleys(intersection)
		Find finger valleys in intersection image:
			- Dilate intersection image to merge incomplete objects.
			- Remove objects around the wrist.
			- Keep smallest 3 remaining objects: the finger valleys.
		Only used within the context of the Bosphorus database.

	calculate_hand_angle(cleaned_image, finger_valleys)
		Calculate the orientation of the hand:
			- Find center of mass of 3 finger valleys as points.
			- Find the middle valley.
			- Find center of mass of fingers_removed image as a point.
			- Find the least square solution between 2 points.
			- The angle is then the degrees of the arctan of the least square solution.
		Only used within the context of the Bosphorus database.

	remove_dark_outline(cleaned_image)
		Remove noise around the boundary of the hand. Only used within the context of the Bosphorus database.

	vertical_alignment(image, angle)
		Wrapper function to inversely rotate image by angle. Only used within the context of the Bosphorus database.

	remove_thumb(removed_outline)
		Function to remove the thumb:
			- Remove the padding applied earlier.
			- Find the center of mass of removed_padding image.
			- Skeletonize the boundary of the eroded object.
			- Find the height of the resulting object.
			- Morphological opening of the resulting object with a vertical line of length height/2.
		Only used within the context of the Bosphorus database.

	remove_wrist(removed_thumb)
		Function to remove the wrist and remainder of the four fingers:
			- Find the center of mass of removed_thumb.
			- Skeletonize the boundary of the eroded object
			- Find the width of the resulting object
			- Morphological opening of the resulting object with a disk of diameter width/2.
		Only used within the context of the Bosphorus database.
	
	find_data_in_ROI(wrist_unpadded, rotated_orig):
		Find the original information in the binary ROI object. 
		Only used within the context of the Bosphorus database.

	remove_boundary_noise(region_of_interest):
		Create an eroded ROI that will be used as a mask to remove boundary noise 
		after each step in the binarisation process. Only used within the context of the Bosphorus database.

	bounding_box_bosphorus(region_of_interest, image):
		Extract a 50x40 bounding box from image around the center of mass of region_of_interest.
		Only used within the context of the Bosphorus database.

	bosphorus_pipeline(original_image, input_image_name):
		Orchestrator function to extract the ROI, CLAHE-enhanced ROI and binarized ROI from an image of the Bosphorus database.

	wilches_pipeline(original_image, input_image_name):
		Orchestrator function to extract the ROI, CLAHE-enhanced ROI and binarized ROI from an image of the Wilches database.

	execute_main()
		Orchestrator function to kick off preprocessing of the input images. Called upon instantiation.

	"""
	def __init__(self, net, base_dir, database):
		self.net = net
		self.base_dir = base_dir
		self.database = database
		self.execute_main()


	def is_preprocessing_complete(self):
		"""
		Checks whether all 3 steps of preprocessing are completed for the given database.
		:return: True if preprocessing completed else False.
		"""
		num_bos = 1200  # number of images in the Bosphorus dataset
		num_wil = 400  # number of images in the Wilches dataset
		comp = 0
		for prep in ['Grey', 'GreyContrast', 'Binary']:
			output_location = self.base_dir+"PhD_Files/Input/{}/{}/".format(self.database, prep)
			if os.path.exists(output_location):
				files = os.listdir(output_location)
				if self.database == 'Bosphorus':
					if len(files) == num_bos:
						comp += 1
				else:
					if len(files) == num_wil:
						comp += 1
			else:
				return False
		if comp != 3:
			return False
		return True


	def set_input_location(self):
		"""
		Set the location of the original images of the given database.
		"""
		self.input_location = self.base_dir+"PhD_Files/Input/{}/Grey/".format(self.database)


	def read_input_image(self, input_location):
		"""
		Read input image of the given database in grayscale format.
		:param input_location: Location of the input image on hard disk.
		:return: 2D numpy array containing the image information.
		"""
		return _io.imread(input_location, as_gray=True)

	
	def save_image(self, name, prep, image):
		"""
		Save preprocessed image in correct location. If location does not exist yet, it is created.
		:param name: Input image filename.
		:param image: Image to save.
		:param prep: Preprocessing step.
		:return: None
		"""
		output_location = self.base_dir+"PhD_Files/Input/{}/{}/".format(self.database, prep)
		if not os.path.exists(output_location):
			os.makedirs(output_location)
		plt.imsave(output_location + name.replace('.bmp', '.png'), image, cmap=plt.get_cmap('gray'))


	def apply_gaussian_smoothing(self, image, factor):
		"""
		Apply gaussian smoothing to the input image.
		:param image: Original input image from the given database.
		:param factor: Smoothing factor.
		:return: Gaussian smoothed image.
		"""
		return gaussian(image, factor)


	def threshold(self, image):
		"""
		Apply Otsu's threshold to the input image.
		:param image: Gaussian smoothed image.
		:return: Binary hand image.
		"""
		return image > threshold_otsu(image)


	def convex_hull(self, image):
		"""
		Find the convex hull of the object in the binary input image.
		:param image: Binary hand image.
		:return: Convex hull image of input image.
		"""
		return convex_hull_image(image)


	def center_of_mass(self, image):
		"""
		Find the center of mass of the object in the binary input image.
		:param image: Convex hull image.
		:return: Coordinates of the center of mass of the convex hull.
		"""
		return np.array(measurements.center_of_mass(image)).astype(int)


	def extract_roi_wilches(self, original_im, convex_hull_im, center_of_mass):
		"""
		The region of interest is defined as the largest rectangle that fits 
		into the convex hull. The approximate aspect ratio of the ROI is 0.8
		in order to correspond to the aspect ratio of the Bosphorus ROI,
		and is centered on the center of mass of the convex hull.
		Only used within the context of the Wilches database.
		:param original_im: Original input image.
		:param convex_hull_im: Convex hull image.
		:param center_of_mass: Center of mass coordinates of convex_hull_im.
		:return: Coordinates of the center of mass of the convex hull.
		"""
		for i in range(30, 100):
			j = int(0.8*i)
			box = convex_hull_im[center_of_mass[0]-i:center_of_mass[0]+i, 
								center_of_mass[1]-j:center_of_mass[1]+j]
			if 0 in box:
				roi = original_im[center_of_mass[0]-(i-1):center_of_mass[0]+(i-1), 
								center_of_mass[1]-(j-1):center_of_mass[1]+(j-1)]
				break
		return roi


	def apply_clahe(self, image, size, limit):
		"""
		Apply CLAHE to the input image.
		:param image: Input (ROI) image.
		:return: CLAHE-enhanced ROI image.
		"""
		return equalize_adapthist(image, kernel_size=np.array([size, size]), clip_limit=limit)
		

	def adaptive_threshold(self, image):
		"""
		Adaptive (local) thresholding to find the mask for morphological reconstruction.
		:param image: Image containing the ROI.
		:return: Locally thresholded ROI image.
		"""
		return 1-(image > threshold_local(image, 9))


	def laplacian(self, image):
		"""
		Laplacian has strong responses near intensity slopes, steps and isolated pixels. 
		This is the first seed image for morphological reconstruction.
		:param image: Image containing the ROI.
		:return: Result of applying the laplacian to the ROI image.
		"""
		return np.abs(filters.laplace(image))

		
	def black_top_hat(self, image):
		"""
		The bottom hat transformation detects narrow dark structures, which is ideal for identifying vein pixels.
		This is the second image for morphological reconstruction.
		:param image: Image containing the ROI.
		:return: Result of applying the morphological black tophat transformation to the ROI image.
		"""
		return grey.black_tophat(image, selem=selem.disk(2))


	def intersection(self, seed1, seed2):
		"""
		Find the morphological skeleton of the intersection of the 2 obtained seed images.
		:param seed1: Thresholded laplacian image of the ROI.
		:param seed2: Thresholded black tophat image of the ROI.
		:return: Skeletonised intersection of seed1 and seed2.
		"""
		return _skeletonize.skeletonize(np.multiply(seed1,seed2))


	def reconstruct(self, seed, mask):
		"""
		Find the morphological reconstruction of the final seed image, constrained by the mask image. First,
		we make sure the seed image contains no white pixels where the mask does not in order to avoid reconstruction
		errors. Secondly, there are a small number of cases where the seed and mask image do not overlap at all. In these
		cases, we are forced to return the mask image as the final binarised version.
		:param seed: Skeletonised intersection of thresholded laplacian and thresholded black tophat images.
		:param mask: Adaptive thresholded ROI.
		:return: Image containing the morphologically reconstructed hand veins.
		"""
		for i in range(seed.shape[0]):
			for j in range(seed.shape[1]):
				if mask[i, j] == 0:
					seed[i, j] = 0
		rec = reconstruction(seed, mask, method='dilation', selem=np.ones((3,3)))
		count = 0
		for i in range(rec.shape[0]):
			for j in range(rec.shape[1]):
				if rec[i, j] > 0:
					count += 1
		if count/float((50*40)) < 0.2:
			return mask
		return rec


	def resize_for_network(self, image):
		"""
		Resizes the final binarized hand vein images to the expected size by the neural networks. 
		Only applicable within the context of the Wilches database, since for the Bosphorus database
		the final images are already correct size.
		:param image: Image containing the morphologically reconstructed hand veins.
		:return: Image containing the morphologically reconstructed hand veins resized to 50 x 40.
		"""
		return resize(image, (50, 40), anti_aliasing=False)


	def erosion(self, image, structure):
		"""
		Wrapper function for binary erosion. Only used within the context of the Bosphorus database.
		:param image: Image to be eroded.
		:param structure: Morphological structuring element used to do the erosion.
		:return: Binary eroded image.
		"""
		return binary.binary_erosion(image, selem=structure)


	def unify_background(self, threshold_im, convex_hull_im):
		"""
		Due to the image acquisition protocol, there is a circle around the hand
		after the application of Otsu's thresholding, which must be unified with the
		actual background of the hand. Only used within the context of the Bosphorus database.
		:param threshold_im: Result of applying Otsu's threshold to the original image.
		:param convex_hull_im: Convex hull of a binary erosion of threshold_im.
		:return: Binary hand with unified background.
		"""
		unified_image = np.copy(threshold_im)
		for i in range(convex_hull_im.shape[0]):
			for j in range(convex_hull_im.shape[1]):
				if convex_hull_im[i,j] == 0:
					unified_image[i,j] = 1
		return unified_image


	def clean(self, unified_image):
		"""
		Remove leftover objects from the background and hand. Pad the 
		unified image with background values in order to ensure the hand
		is fully contained in the image. Only used within the context of the Bosphorus database.
		:param unified_image: Binary hand with unified background.
		:return: unified_image with noise removed.
		"""
		cleaned_image = np.copy(unified_image)
		cleaned_image = np.pad(cleaned_image, 5, constant_values=1)
		labeled, _ = lb.label(cleaned_image, Bc=np.ones((3,3)))
		for i in range(labeled.shape[0]):
			for j in range(labeled.shape[1]):
				if labeled[i,j] > 1:
					cleaned_image[i,j] = 0
		return 1-cleaned_image


	def remove_fingers(self, cleaned_image):
		"""
		Remove the four fingers with binary opening. Only used within the context of the Bosphorus database.
		:param cleaned_image: Binary hand image.
		:return: cleaned_image with four fingers removed.
		"""
		return binary.binary_opening(cleaned_image, selem=selem.disk(15))


	def find_hand_boundary(self, cleaned_image):
		"""
		Find the hand boundary. This will be used to identify the intersection
		points between the fingers, which will subsequenctly be used to calculate the
		orientation of the hand. Only used within the context of the Bosphorus database.
		:param cleaned_image: Binary hand image.
		:return: Boundary image of cleaned_image.
		"""
		return cleaned_image - self.erosion(cleaned_image, selem.disk(1))


	def overlay(self, fingers_removed, hand_boundary):
		"""
		Find the intersection between the binary hand with fingers removed and the hand boundary.
		Only used within the context of the Bosphorus database.
		:param fingers_removed: Binary hand image with fingers removed.
		:param hand_boundary: Boundary of binary hand.
		:return: Intersection between fingers_removed and hand_boundary.
		"""
		return np.multiply(fingers_removed, hand_boundary)
		

	def find_finger_valleys(self, intersection):
		"""
		Find finger valleys in intersection image:
			- Dilate intersection image to merge incomplete objects.
			- Remove objects around the wrist.
			- Keep smallest 3 remaining objects: the finger valleys.
		Only used within the context of the Bosphorus database.
		:param intersection: Intersection image between fingers removed and hand boundary.
		:return: Image containing 3 finger valleys.
		"""
		intersection = morphology.binary_dilation(intersection, structure = np.ones((5,5)), iterations=2)

		# This part cleans the wrist area
		labeled, number_of_objects = measurements.label(intersection,structure=np.ones((3,3)))
		labels = np.arange(1,number_of_objects+1)
		for label in labels:
			temp = np.zeros(intersection.shape)
			for i in range(labeled.shape[0]):
				for j in range(labeled.shape[1]):
					if labeled[i,j] == label:
						temp[i,j] = 1
			cam = self.center_of_mass(temp)
			if np.abs(intersection.shape[0] - cam[0]) < 50:
				for m in range(labeled.shape[0]):
					for n in range(labeled.shape[1]):
						if labeled[m,n] == label:
							intersection[m,n] = 0

		# And this part keeps only the 3 smallest objects remaining: the areas of the finger valleys.
		labeled1, _ = lb.label(intersection,Bc=np.ones((3,3)))
		sizes =lb.labeled_size(labeled1)
		ranks = np.sort(sizes)
		labels_to_keep = []
		for size in ranks[:3]:
			labels_to_keep.append(list(sizes).index(size))
			sizes[list(sizes).index(size)] = 50000
		out = np.zeros(intersection.shape)
		for c in range(labeled1.shape[0]):
			for d in range(labeled1.shape[1]):
				if labeled1[c,d] in labels_to_keep:
					out[c,d] = 1
		return out


	def calculate_hand_angle(self, cleaned_image, finger_valleys):
		"""
		Calculate the orientation of the hand:
			- Find center of mass of 3 finger valleys as points.
			- Find the middle valley.
			- Find center of mass of fingers_removed image as a point.
			- Find the least square solution between 2 points.
			- The angle is then the degrees of the arctan of the least square solution.
		Only used within the context of the Bosphorus database.
		:param cleaned_image: Binary hand with fingers removed.
		:param finger_valleys: Image containing 3 finger valleys.
		:return: None
		"""		
		# This extracts the center of mass of each of the 3 valleys as coordinates
		labeled1, number_of_objects1 = measurements.label(finger_valleys, structure=np.ones((3,3)))
		labels1 = np.arange(1,number_of_objects1+1)
		valleys = []
		for label in labels1:
			temp = np.zeros(finger_valleys.shape)
			for o in range(labeled1.shape[0]):
				for p in range(labeled1.shape[1]):
					if labeled1[o,p] == label:
						temp[o,p] = 1
			cam = measurements.center_of_mass(temp)
			valleys.append(cam)

		# The reference point image, and the coordinates thereof.
		a = np.zeros(finger_valleys.shape)
		refs = []

		# We want only the middle reference point.
		ys = []
		for cma in valleys:
			ys.append(cma[1])
		ys = np.sort(ys)
		for cd in valleys:
			if ys[1] in cd:
				refs.append(cd)
				a[int(cd[0]),int(cd[1])] = 1

		# The center of mass is the second reference point.
		handcam = measurements.center_of_mass(cleaned_image)
		refs.append(handcam)
		a[int(handcam[0]),int(handcam[1])] = 1
		
		A = [[1, refs[0][0]],
			[1, refs[1][0]]]
		b = [refs[0][1],
			refs[1][1]]
		sol, _, _, _ = ln.lstsq(A,b)
		return np.round(np.degrees(np.arctan(sol[1])))


	def remove_dark_outline(self, cleaned_image):
		"""
		Remove noise around the boundary of the hand. Only used within the context of the Bosphorus database.
		:param cleaned_image: Binary hand with fingers removed.
		:return: cleaned_image with boundary noise removed.
		"""
		return morphology.binary_erosion(cleaned_image, structure=np.ones((3,3)),iterations=3)


	def veritical_alignment(self, image, angle):
		"""
		Wrapper function to inversely rotate image by angle. Only used within the context of the Bosphorus database.
		:param image: Image to rotate.
		:param angle: Angle of object with respect to a vertical line.
		:return: image rotated upright.
		"""
		return _warps.rotate(image, -float(angle))


	def remove_thumb(self, removed_outline):
		"""
		Function to remove the thumb:
			- Remove the padding applied earlier.
			- Find the center of mass of removed_padding image.
			- Skeletonize the boundary of the eroded object.
			- Find the height of the resulting object.
			- Morphological opening of the resulting object with a vertical line of length height/2.
		Only used within the context of the Bosphorus database.
		:param removed_outline: Cleaned ROI with boundary removed.
		:return: removed_outline with thumb cut off due to opening.
		"""
		removed_outline = removed_outline[5:245, 5:305]
		cam = measurements.center_of_mass(removed_outline)
		er = morphology.binary_erosion(removed_outline, structure=np.ones((3,3)), iterations=8)
		bound = _skeletonize.skeletonize(removed_outline - er)
		heightcoords = []
		for x in range(removed_outline.shape[0]):
			for y in range(removed_outline.shape[1]):
				if y == int(cam[1]) and bound[x,y] == 1:
					heightcoords.append((x,y))
		height = int(np.abs(heightcoords[0][0] - cam[0]))
		remove_thumb = np.ones((height,1))
		return morphology.binary_opening(removed_outline, structure=remove_thumb)


	def remove_wrist(self, removed_thumb):
		"""
		Function to remove the wrist and remainder of the four fingers:
			- Find the center of mass of removed_thumb.
			- Skeletonize the boundary of the eroded object
			- Find the width of the resulting object
			- Morphological opening of the resulting object with a disk of diameter width/2.
		Only used within the context of the Bosphorus database.
		:param removed_thumb: Resultant image after thumb removal.
		:return: removed_thumb with wrist and four fingers removed.
		"""
		cam2 = measurements.center_of_mass(removed_thumb)
		er2 = morphology.binary_erosion(removed_thumb, structure=np.ones((3,3)), iterations=3)
		bound2 = _skeletonize.skeletonize(np.bitwise_xor(removed_thumb, er2))
		widthcoords = []
		for x1 in range(removed_thumb.shape[0]):
			for y1 in range(removed_thumb.shape[1]):
				if x1 == int(cam2[0]) and bound2[x1,y1] == 1:
					widthcoords.append((x1,y1))
		width = int(np.abs(widthcoords[0][1] - widthcoords[1][1]))
		return morphology.binary_opening(removed_thumb, structure=selem.disk(int(width/2)))


	def find_data_in_ROI(self, wrist_unpadded, rotated_orig):
		"""
		Find the original information in the binary ROI object. 
		Only used within the context of the Bosphorus database.
		:param wrist_unpadded: Rotated ROI coordinates.
		:param rotated_orig: Rotated original image.
		:return: Rotated ROI with original image information.
		"""
		return np.multiply(wrist_unpadded, rotated_orig)


	def remove_boundary_noise(self, region_of_interest):
		"""
		Create an eroded ROI that will be used as a mask to remove boundary noise 
		after each step in the binarisation process. Only used within the context of the Bosphorus database.
		:param region_of_interest: Rotated ROI with original image information.
		:return: Eroded region_of_interest coordinates.
		"""
		hull = np.zeros(region_of_interest.shape)
		for i in range(region_of_interest.shape[0]):
			for j in range(region_of_interest.shape[1]):
				if region_of_interest[i, j] > 0:
					hull[i, j] = 1
		return binary.binary_erosion(hull, selem=selem.disk(11))


	def bounding_box_bosphorus(self, region_of_interest, image):
		"""
		Extract a 50x40 bounding box from image around the center of mass of region_of_interest.
		Only used within the context of the Bosphorus database.
		:param region_of_interest: Used to find the center of mass. 
		:param image: Image with same shape as region_of_interest from which to extract the bounding box.
		:return: 50x40 bounding box around the center of mass of image.
		"""
		cam = self.center_of_mass(region_of_interest)
		return image[cam[0]-25:cam[0]+25,cam[1]-20:cam[1]+20]


	def genveins_pipeline(self, original_image, input_image_name):
		"""
		Orchestrator function to run preprocessing steps for an image in the artificially
		generated GenVeins database.
		:param original_image: Input (greyscale ROI) image from GenVeins database.
		:param input_image_name: Name of input image for output file naming.
		:return: None
		"""
		
		smoothed_roi = 1-original_image

		# GreyContrast
		clahe = self.apply_clahe(smoothed_roi, 5, 0.03)
		resized_clahe = 1-self.resize_for_network(clahe)
		self.save_image(input_image_name, 'GreyContrast', resized_clahe)

		# Binary
		smoothed_roi = self.apply_gaussian_smoothing(smoothed_roi, 1.5)
		mask = self.adaptive_threshold(smoothed_roi)
		lapl = self.laplacian(smoothed_roi)
		seed1 = self.threshold(lapl)
		blth = self.black_top_hat(smoothed_roi)
		seed2 = self.threshold(blth)
		seed3 = self.intersection(seed1, seed2)
		reconstructed = self.reconstruct(seed3, mask)
		self.save_image(input_image_name, 'Binary', reconstructed)


	def execute_main(self):
		"""
		Orchestrator function to kick off preprocessing of the input images. Called upon instantiation.
		:return: None
		"""
		self.set_input_location()
		input_images = os.listdir(self.input_location)

		for input_image_name in input_images:
			
			image_location = self.input_location+input_image_name
			original_image = self.read_input_image(image_location)
			self.genveins_pipeline(original_image, input_image_name)

VeinsNetDataPreprocessor('Siam', 'D:', 'GenVeins')