# System imports
import sys
import warnings
from itertools import product

# Third party imports
import numpy as np

# Local imports
from utils import *

# Increase python default recursion limit to enable
# search for acceptable hand vein image.
sys.setrecursionlimit(100000)

# Supress irrelevant warnings
warnings.filterwarnings("ignore", category=UserWarning)


def generate_trunk_vein_seed_pairs():
    """
    Generate start and end points for trunk veins. The proposed algorithm is as follows:
            - Seed pairs are located on different edges of the image. Favour horizontal, rather than vertical edges
                    due to the typical direction of seed veins inside a hand.
            - Number of seed pairs will be either 2 or 3 (resulting in 2 or 3 trunk veins - favour 2 trunk veins).
            - Choose initial seed points for each trunk vein and verify the following 2 conditions (based on image size 50 x 40):
                    ~ Squared Euclidean Distance between seed points must be larger or equal to 35 (span enough of the image).
                    ~ Vertical distance between seed points must be larger or equal to 40 (Ensure vertically oriented seed veins).
            - Repeat for number of seed point pairs, while only adding another seed point pair if verify_trunk_vein() is True.
                    See method docstring for conditions.

    Returns:
    --------
    list
            Seed point pairs for trunk veins.
    """
    num_seed_points = np.random.choice([2, 3], p=[0.65, 0.35])
    image_border_ranges = np.array(
        [
            [(0, n) for n in range(0, 40)],
            [(m, 39) for m in range(0, 50)],
            [(49, n) for n in range(0, 40)],
            [(m, 0) for m in range(0, 50)],
        ],
        dtype=object,
    )
    seed_point_pairs = []
    for _ in range(num_seed_points):
        borders = np.random.choice(
            image_border_ranges, size=2, replace=False, p=[0.3, 0.2, 0.3, 0.2]
        )
        seed_1 = borders[0][np.random.randint(len(borders[0]))]
        seed_2 = borders[1][np.random.randint(len(borders[1]))]
        while (
            dist_between_seeds(seed_1, seed_2) < 35
            or i_dist_between_seeds(seed_1, seed_2) < 40
        ):
            seed_1 = borders[0][np.random.randint(len(borders[0]))]
            seed_2 = borders[1][np.random.randint(len(borders[1]))]
        if len(seed_point_pairs) > 0:
            while not verify_trunk_vein(seed_point_pairs, seed_1, seed_2):
                borders = np.random.choice(
                    image_border_ranges, size=2, replace=False, p=[0.3, 0.2, 0.3, 0.2]
                )
                seed_1 = borders[0][np.random.randint(len(borders[0]))]
                seed_2 = borders[1][np.random.randint(len(borders[1]))]
                while (
                    dist_between_seeds(seed_1, seed_2) < 35
                    or i_dist_between_seeds(seed_1, seed_2) < 40
                ):
                    seed_1 = borders[0][np.random.randint(len(borders[0]))]
                    seed_2 = borders[1][np.random.randint(len(borders[1]))]
        seed_point_pairs.append(sorted([seed_1, seed_2]))
    return seed_point_pairs


def generate_branch_vein_seed_pairs(trunk_veins):
    """
    Generate start and end points for branch veins. The proposed algorithm is as follows:
            - Find trunk vein coordinates. Branch veins invariably run between any two different trunk veins
            - Randomly select number of branch veins from [0, 1, 2, 3], favouring 1 and 2.
            - Branch veins are encouraged to flow in a vertical, rather than horizontal direction.

    Parameters:
    -----------
    trunk_veins: numpy.array
            Image from which to determine branch seed point pairs.

    Returns:
    --------
    list
            Seed point pairs for branch veins.
    """
    vein_coords, _ = find_coords(trunk_veins)
    num_branch_seeds = np.random.choice([0, 1, 2, 3], p=[0.2, 0.35, 0.3, 0.15])
    branch_seed_pairs = []
    if not num_branch_seeds == 0:
        for _ in range(num_branch_seeds):
            seed_1 = vein_coords[np.random.randint(len(vein_coords))]
            seed_2 = vein_coords[np.random.randint(len(vein_coords))]
            while i_dist_between_seeds(seed_1, seed_2) < 20:
                seed_1 = vein_coords[np.random.randint(len(vein_coords))]
                seed_2 = vein_coords[np.random.randint(len(vein_coords))]
            branch_seed_pairs.append(sorted([seed_1, seed_2]))
    return branch_seed_pairs


def verify_trunk_vein(seed_point_pairs, seed_1, seed_2):
    """
    Check that seed pairs associated with trunk veins adhere to the following constraint.
            - The difference between angles of all trunk veins should be large enough.
              This is an attempt to mitigate the problem of adjacent seed veins
              (as it is rarely the case with actual hand veins).

    Parameters:
    -----------
    seed_point_pairs: list
            Exisiting seed point pairs from which to determine angle between new seed point pair.

    seed_1: tuple
            New seed point 1.

    seed_2: tuple
            New seed point 2.

    Returns:
    --------
    bool
            True if angle between new seed point pair is different enough from all existing seed point pairs and False otherwise.
    """
    angles = []
    for spp in seed_point_pairs:
        angles.append(angle_between_seeds(spp[0], spp[1]))
    new_angle = angle_between_seeds(seed_1, seed_2)
    angle_prod = list(product([new_angle], angles))
    accepted_angle_range = [v for v in range(15, 45)]
    if all([abs(v[0] - v[1]) in accepted_angle_range for v in angle_prod]):
        return True
    return False


def verify_vein_spread(im):
    """
    Verify that the random propagation algorithm introduced enough spread of the veins across the image.
    If not, image is discarded and a new one is created in its place until this function returns True.

    Parameters:
    -----------
    im: numpy.array
            Sample trunk vein image from which to calculate spread.

    Returns:
    --------
    bool
            False if percentage of white pixels fall out of the given ranges.
    """
    spread = np.count_nonzero(im) / 2000.0  # 50 x 40 = 2000
    if spread < 0.35 or spread > 0.70:
        return False
    return True


def draw_trunk_veins():
    """
    Draw seed veins on a black image. The proposed algorithm is as follows:
            - Generate seed point pairs for trunk veins based.
            - Draw acceptable trunk veins.
            - Dilate seed veins by an acceptable factor (seed veins are typically thicker than branch and other veins).

    Returns:
    --------
    numpy.array
            Black image with added trunk veins.
    """
    im = initialize_im()
    seed_point_pairs = generate_trunk_vein_seed_pairs()
    im = propagate_and_draw_veins(im, seed_point_pairs)
    im = dilate(im, 2)
    return im


def draw_branch_veins(trunk_veins):
    """
    Draw branch veins on a black image based on the trunk veins. The proposed algorithm is as follows:
            - Branch veins run from one trunk vein to another.
            - They follow the same propagation algorithm as generating the trunk veins.
            - Dilate branch veins by an acceptable factor (typically thinner than seed veins).

    Parameters:
    -----------
    trunk_veins: numpy.array
            Image containing trunk veins from which to determine and draw branch veins.

    Returns:
    --------
    numpy.array
            Black image with added branch veins.
    """
    im = initialize_im()
    branch_point_pairs = generate_branch_vein_seed_pairs(trunk_veins)
    if len(branch_point_pairs) > 0:
        im = propagate_and_draw_veins(im, branch_point_pairs)
        im = dilate(im, 1.5)
    return im


def union_base_vein_ims(trunk_veins, branch_veins):
    """
    Union the generated trunk and branch hand vein images.

    Parameters:
    -----------
    trunk_veins: numpy.array
            Trunk vein image.

    branch_veins: numpy.array
            Branch vein image.

    Returns:
    --------
    numpy.array
            Base vein image.
    """
    base_veins = initialize_im()
    for i in range(base_veins.shape[0]):
        for j in range(base_veins.shape[1]):
            if trunk_veins[i, j] > 0 or branch_veins[i, j] > 0:
                base_veins[i, j] = 1
    return base_veins


def main_function(root_output_dir, ind):
    """
    Orchestrator function for generating base (trunk + branch) veins. The proposed algorithm is as follows:
            - Draw the first iteration of trunk and branch veins on a black background.
            - Verify the spread of obtained base veins are acceptable and continue to restart until this is so.

    Parameters:
    -----------
    root_output_dir: str
            Location to save the resultant images.

    ind: int
            Numbered individual for which base veins are being generated.

    Returns:
    --------
    numpy.array
            Binary image containing base veins.
    """
    trunk_veins = draw_trunk_veins()
    branch_veins = draw_branch_veins(trunk_veins)
    base_veins = union_base_vein_ims(trunk_veins, branch_veins)
    while verify_vein_spread(base_veins) == False:
        trunk_veins = draw_trunk_veins()
        branch_veins = draw_branch_veins(trunk_veins)
        base_veins = union_base_vein_ims(trunk_veins, branch_veins)
    save(root_output_dir, base_veins, ind, "", "Base_Veins")
    return base_veins
