# System imports
import multiprocessing
import sys
import time
from functools import partial

# Local imports
from structural_variation_generator import main_function as struct_main
from greyification import main_function as back_main
from base_generator import main_function as base_main


def main_function(ind, num_ims, root_output_dir):
    """
    Orchestrator function for the repo. The proposed algorithm is briefly outlined below, while an in depth
    explanation of each step is documented in the corresponding module and functions.
            - Start by drawing the base veins on a black image. These constitute trunk and branch veins (see base_generator.py).
            - Next, create num_ims copies of the base veins and add tree veins and unconnected veins (see structural_variation_generator.py).
            - Lastly, greyify each of the copies (see greyification.py).

    Parameters:
    -----------
    ind: int
            Numbered fictitious individual for which to generate hand veins.

    num_ims: int
            Number of images to generate for each fictitious individual.

    root_output_dir: str
            Location to save generated images.
    """
    base_veins = base_main(root_output_dir, ind)
    struct_veins = struct_main(root_output_dir, base_veins, ind, num_ims)
    back_main(root_output_dir, struct_veins, ind)
    print("Individual {} done.".format(ind))


def generate_database(num_inds, num_ims, root_output_dir):
    """
    Wrapper function for multiprocessing main_function with fixed arguments.

    Parameters:
    -----------
    num_inds: int
            Number of fictitious individuals for which to generate hand veins.

    num_ims: int
            Number of images to generate for each fictitious individual.

    root_output_dir: str
            Location to save generated images.
    """
    # Use functools.partial to fix num_ims and root_output_dir arguments for main_function
    partial_main_function = partial(
        main_function, num_ims=num_ims, root_output_dir=root_output_dir
    )
    num_inds_list = [v for v in range(num_inds)]
    num_cpus = multiprocessing.cpu_count()
    start = time.time()
    with multiprocessing.Pool(processes=num_cpus) as pool:
        results = pool.map_async(partial_main_function, num_inds_list)
        _ = results.get()
        pool.close()
        pool.join()
    print(
        "Generating GenVeins database with {} individuals complete in {} seconds".format(
            num_inds, time.time() - start
        )
    )


if __name__ == "__main__":
    """
    Refer to "Usage" section in README.md for explanations on these parameters. They are required for each execution.
    """
    num_inds = int(sys.argv[1])
    num_ims = int(sys.argv[2])
    root_output_dir = sys.argv[3]
    generate_database(num_inds, num_ims, root_output_dir)
