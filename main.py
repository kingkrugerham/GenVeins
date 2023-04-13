# System imports
import multiprocessing
import sys
import time

# Local imports
from structural_variation_generator import main_function as struct_main
from greyification import main_function as back_main
from base_generator import main_function as base_main


def main_function(ind):
	"""
	Orchestrator function for the repo. The proposed algorithm is briefly outlined below, while an in depth
	explanation of each step is documented in the corresponding module and functions.
		- Start by drawing the base veins on a black image. These constitute seed and branch veins.
		- Next, create 4 copies of the base veins and add tree veins and unconnected veins (simulate acquisition noise).
		- Lastly, greyify each of the 4 copies.
	"""
	root_output_dir = f'PhD_Files/Input/GenVeinsV5/'
	base_veins = base_main(root_output_dir, ind)
	struct_veins = struct_main(root_output_dir, base_veins, ind)
	back_main(root_output_dir, struct_veins, ind)
	print('Individual {} done.'.format(ind))


if __name__ == "__main__":
	start = time.time()
	num_inds = int(sys.argv[1])  # number of individuals in the database to be generated.
	num_inds_list = [v for v in range(num_inds)]
	num_cpus = multiprocessing.cpu_count()
	with multiprocessing.Pool(processes=num_cpus) as pool:
		results = pool.map_async(main_function, num_inds_list)
		_ = results.get()
		pool.close()
		pool.join()
	print('Generating GenVeinsV5 database with {} individuals complete in {} seconds'.format(num_inds, time.time()-start))
