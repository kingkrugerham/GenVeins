# veinsnet_dcgan
Repository for investigative methods to generate an artificial hand vein database. 

Methods explored that didn't work (see ./non_feasible_solutions):

	1 - Deep Convolutional GAN trained on existing database (Bosphorus database).
			Produces samples that look too much like the samples within the database trained on. Possibly
			because only 100 different structures are available in said database.
	2 - Fully Connected (Simple Linear) GAN.
			Same outcome as DCGAN
	3 - Intersection of random pairs of veins in existing database.
			Results do not look like hand veins.

Current working method for generating four artificial hand vein images for a specific fictitious individual:

	1 - One unique sample of so-called *base veins* is first generated, which represents the main vein structure of a fictitious individual.
	2 - Four copies of each unique sample are subsequently created, after which each copy is supplemented by randomly adding so-called  
	 	*auxiliary veins*. The purpose of this step is to simulate minor structural differences that arise during the acquisition 
		of multiple samples of the hand veins belonging to an actual individual.
	3 - Each supplemented copy, which constitutes a binary image, is then "greyed" in order to obtain four non-binary regions of interest 
		(ROIs) for each artificial individual. The purpose of this step is to obtain samples similar to the ROIs of dimensions 50 x 40 
		pixels.
	4 - Each artificial non-binary ROI is finally subjected to the same preprocessing protocol as the one proposed in 
		https://doi.org/10.5281/zenodo.6961864.

The full report on this repository and protocols within is published in https://doi.org/10.5281/zenodo.6961864 (REPLACE WHEN PUBLISHED)