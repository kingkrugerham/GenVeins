# veinsnet_dcgan
Repository for investigative methods to generate an artificial hand vein database. 

Methods explored that didn't work:

	1 - Deep Convolutional GAN trained on existing database.
			Produces samples that look too much like the samples within the database trained on. Possibly
			because only 100 different structures are available in said database.
	2 - Fully Connected (Simple Linear) GAN.
			Same outcome as DCGAN
	3 - Intersection of random pairs of veins in existing database.
			Results do not look like hand veins.

Current working method for generating binary hand vein-like structures:

	