GenVeins
===================================================================================
Repository for investigative methods to generate an artificial hand vein database. 

### Methods explored that didn't work (see ./non_feasible_solutions):

	1 - Deep Convolutional GAN trained on existing database (Bosphorus database).
			Produces samples that look too much like the samples within the database trained on. Possibly
			because only 100 different structures are available in said database.
	2 - Fully Connected (Simple Linear) GAN.
			Same outcome as DCGAN
	3 - Intersection of random pairs of veins in existing database.
			Results do not look like hand veins.

### Current working method for generating artificial hand vein images for a specific fictitious individual:

	1 - One unique sample of so-called *base veins* is first generated, which represents the main vein structure of a fictitious individual.
	2 - n copies of each unique sample are subsequently created, after which each copy is supplemented by randomly adding so-called  
	 	*auxiliary veins*. The purpose of this step is to simulate minor structural differences that arise during the acquisition 
		of multiple samples of the hand veins belonging to an actual individual.
	3 - Each supplemented copy, which constitutes a binary image, is then "greyed" in order to obtain four non-binary regions of interest 
		(ROIs) for each artificial individual. The purpose of this step is to obtain samples similar to the ROIs of dimensions 50 x 40 
		pixels.
	4 - Each artificial non-binary ROI is finally subjected to the same preprocessing protocol as the one proposed in 
		https://doi.org/10.5281/zenodo.6961864.

The full report on this repository and protocols within is published in https://doi.org/10.5281/zenodo.6961864 (REPLACE WHEN PUBLISHED)

### Usage:
	1 - Entry point is main.py
	2 - sys.argv[1] : Number of individuals to generate.
	3 - sys.argv[2] : Number of images to generate for each individual.
	4 - sys.argv[3] : Output directory where images get saved.


### Example: 
Generate 4 artificial hand vein images of shape 50 x 40 (i, j) for each of 10 fictitious individuals and save in folder 'GenVeinsV1' inside
the root directory.
```
python main.py 10 4 GenVeinsV1
```

### Changes to protocol to fit user use case:
Currently generating images of shape 50 x 40. A lot of the 'empirical' values set in this repo has been done through trial and error in order to avoid things such as infinte loops while maintaining good quality generated samples. The recalibration therefore has to be done again through trial and error by the user in order to accommodate their use case. This can be done in the following steps:

	1 - First modify the base_generator.py and get working base veins for your specific use case
	2 - Then modify structural_variation_generator.py which creates copies of the base vein structure
	3 - No work required within the greyification.py script when only image dimensions have changed, but feel free to play around with 
		different levels of smoothing etc.

### Additional notes:
	1 - Repo formatted with black which uses space indentation (https://black.readthedocs.io/en/stable/usage_and_configuration/index.html)
	2 - Docstrings formatted according to NumPy-style.
	3 - Code in non_feasible_solutions may not work anymore, since the repo has changed significantly since.
	4 - Some functions are non-optimal, and could be refactored.
	5 - Multiprocessing is used to simultaneously generate hand vein samples for a large number of different individuals.
	6 - It is most likely better to use the grey versions of the images in your process, and apply necessary contrast enhancement,
		rather than using the binary hand vein images as is.