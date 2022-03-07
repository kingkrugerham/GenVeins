
# # def draw_tree_veins(root_output_dir, i, unioned):
# # 	"""
# # 	Add tree veins based on the current vein information. The proposed algorithm is as follows:
# # 		- Tree veins propagate out of existing veins towards another random seed point not within the current vein structure.
# # 		- They follow the same propagation algorithm as generating the seed veins.
# # 	:param root_output_dir: Location to save images.
# # 	:param i: ith image in the list.
# # 	:return: Image with added tree veins.
# # 	"""
# # 	im = initialize_im()
# # 	tree_veins_dir = root_output_dir + '4.AddedTreeVeins/'
# # 	make_output_dir(tree_veins_dir)
# # 	tree_point_pairs = generate_tree_seed_pairs(im, unioned)
# # 	if len(tree_point_pairs) > 0:
# # 		im = propagate_and_draw_veins(im, i, tree_point_pairs)
# # 		dilation_factor = np.random.choice([1, 1.5, 2], p=[0.4, 0.4, 0.2])
# # 		im = dilate(im, dilation_factor)
# # 	plt.imsave(tree_veins_dir + 'person_'+str(i)+'.png', im, cmap=plt.get_cmap('gray'))
# # 	return im


# # def draw_unconnected_veins(root_output_dir, i, unioned):
# # 	"""
# # 	Add unconnected veins based on the current vein information. The proposed algorithm is as follows:
# # 		- Unconnected veins are not connected to the existing vein structure (introduces some noise into the image).
# # 		- They follow the same propagation algorithm as generating the seed veins.
# # 	:param root_output_dir: Location to save images.
# # 	:param i: ith image in the list.
# # 	:return: Image with added tree veins.
# # 	"""
# # 	im = initialize_im()
# # 	unconnected_veins_dir = root_output_dir + '5.AddedUnconnectedVeins/'
# # 	make_output_dir(unconnected_veins_dir)
# # 	unconnected_seed_pairs = generate_unconnected_seed_pairs(im, unioned)
# # 	if len(unconnected_seed_pairs) > 0:
# # 		im = propagate_and_draw_veins(im, i, unconnected_seed_pairs)
# # 		dilation_factor = np.random.choice([1, 1.5, 2], p=[0.4, 0.4, 0.2])
# # 		im = dilate(im, dilation_factor)
# # 	plt.imsave(unconnected_veins_dir + 'person_'+str(i)+'.png', im, cmap=plt.get_cmap('gray'))
# # 	return im


# # def union_vein_ims(seed_veins, branch_veins, tree_veins, uncon_veins):
# # 	"""
# # 	Union all the generated hand vein images.
# # 	:param seed_veins: Seed vein image.
# # 	:param branch_veins: Branch vein image.
# # 	:param tree_veins: Tree vein image.
# # 	:param uncon_veins: Unconnected vein image.
# # 	:return: Final vein image.
# # 	"""
# # 	final_veins = initialize_im()
# # 	for i in range(final_veins.shape[0]):
# # 		for j in range(final_veins.shape[1]):
# # 			if seed_veins[i, j] == 1 or branch_veins[i, j] == 1 or tree_veins[i, j] == 1 or uncon_veins[i, j] == 1:
# # 				final_veins[i, j] = 1
# # 	return final_veins

# def generate_tree_seed_pairs(im, unioned):
# 	"""
# 	Generate a number of tree seed point pairs to connect with vein-like structures.
# 	:param im: Image from which to determine tree seed point pairs.
# 	:return: List of seed point pairs for tree veins.
# 	"""
# 	vein_coords, back_coords = [], []
# 	for i in range(unioned.shape[0]):
# 		for j in range(unioned.shape[1]):
# 			if unioned[i, j] == 1:
# 				vein_coords.append((i, j))
# 			else:
# 				back_coords.append((i, j))

# 	num_tree_seeds = np.random.choice([0, 1, 2], p=[0.35, 0.35, 0.3])
# 	tree_seed_pairs = []
# 	if not num_tree_seeds == 0:
# 		for _ in range(num_tree_seeds):
# 			seed_1 = vein_coords[np.random.randint(len(vein_coords))]
# 			seed_2 = back_coords[np.random.randint(len(back_coords))]
# 			while i_dist_between_seeds(seed_1, seed_2) < 10 or j_dist_between_seeds(seed_1, seed_2) < 10:
# 				seed_1 = vein_coords[np.random.randint(len(vein_coords))]
# 				seed_2 = back_coords[np.random.randint(len(back_coords))]
# 			tree_seed_pairs.append(sorted([seed_1, seed_2]))
# 	return tree_seed_pairs


# def generate_unconnected_seed_pairs(im, unioned):
# 	"""
# 	Generate a number of unconnected seed point pairs to connect with vein-like structures.
# 	:param im: Image from which to determine unconnected seed point pairs.
# 	:return: List of seed point pairs for unconnected veins.
# 	"""
# 	vein_coords, back_coords = [], []
# 	for i in range(unioned.shape[0]):
# 		for j in range(unioned.shape[1]):
# 			if unioned[i, j] == 1:
# 				vein_coords.append((i, j))
# 			else:
# 				back_coords.append((i, j))

# 	num_unconnected_seeds = np.random.choice([0, 1, 2], p=[0.35, 0.35, 0.3])
# 	unconnected_seed_pairs = []
# 	if not num_unconnected_seeds == 0:
# 		for _ in range(num_unconnected_seeds):
# 			seed_1 = back_coords[np.random.randint(len(back_coords))]
# 			seed_2 = back_coords[np.random.randint(len(back_coords))]
# 			while i_dist_between_seeds(seed_1, seed_2) < 10 or j_dist_between_seeds(seed_1, seed_2) < 10:
# 				seed_1 = back_coords[np.random.randint(len(back_coords))]
# 				seed_2 = back_coords[np.random.randint(len(back_coords))]
# 			unconnected_seed_pairs.append(sorted([seed_1, seed_2]))
# 	return unconnected_seed_pairs