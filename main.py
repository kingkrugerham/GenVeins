# System imports
import sys

# Local imports
from structural_variation_generator import main_function as struct_main
from background_generator import main_function as back_main
from base_generator import main_function as base_main


def main_function(root_output_dir, num_inds):
	for ind in range(num_inds):
		base_veins = base_main(root_output_dir, ind)
		struct_veins = struct_main(root_output_dir, base_veins, ind)
		back_main(root_output_dir, struct_veins, ind)

if __name__ == "__main__":
	root_output_dir = 'D:/PhD_Files/Input/GenVeins/'
	num_inds = sys.argv[1]
	main_function(root_output_dir, num_inds)
	