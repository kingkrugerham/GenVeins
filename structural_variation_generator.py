# System imports
import sys

# Third party imports
import cv2
import numpy as np
from skimage import transform as tf

# Local imports
from utils import *

# Increase python default recursion limit to enable
# search for acceptable hand vein image.
sys.setrecursionlimit(100000)


def generate_twig_vein_seed_pairs(base_veins):
    """
    Generate a number of twig vein seed point pairs. The proposed algorithm is as follows:
            - Find vein and background coordinates in original_im.
            - Randomly choose number of twig seed pairs (either 0, 1 or 2) in favour of 1 twig vein.
            - Verify the vertical and horizontal distances between selected twig seed points are in acceptable range.
                    (twig veins are typically much shorter than base veins).
            - At least one of the seed points of twig veins invariably lie on a base (trunk or branch) vein.

    Paramters:
    ----------
    base_veins: numpy.array
            Image containing base veins from which to determine twig seed point pairs.

    Returns:
    --------
    list
            Seed point pairs for twig veins.
    """
    vein_coords, back_coords = find_coords(base_veins)
    num_twig_seeds = np.random.choice([0, 1, 2], p=[0.3, 0.5, 0.2])
    twig_seed_pairs = []
    if not num_twig_seeds == 0:
        for _ in range(num_twig_seeds):
            seed_1 = vein_coords[np.random.randint(len(vein_coords))]
            seed_2 = back_coords[np.random.randint(len(back_coords))]
            accepted_range = np.arange(10, 20)
            while (
                i_dist_between_seeds(seed_1, seed_2) not in accepted_range
                or j_dist_between_seeds(seed_1, seed_2) not in accepted_range
            ):
                seed_1 = vein_coords[np.random.randint(len(vein_coords))]
                seed_2 = back_coords[np.random.randint(len(back_coords))]
            twig_seed_pairs.append(sorted([seed_1, seed_2]))
    return twig_seed_pairs


def generate_unconnected_vein_seed_pairs(base_veins):
    """
    Generate a number of unconnected vein seed point pairs. The proposed algorithm is as follows:
            - Same selection criteria as twig veins.
            - Unconnected veins are not required to be connected to any existing veins.

    Parameters:
    -----------
    original_im: numpy.array:
            Image from which to determine unconnected vein seed point pairs.

    Returns:
    --------
    list
            Seed point pairs for unconnected veins.
    """
    _, back_coords = find_coords(base_veins)
    num_objects = np.random.choice([0, 1, 2], p=[0.3, 0.5, 0.2])
    unconnected_seed_pairs = []
    if not num_objects == 0:
        for _ in range(num_objects):
            seed_1 = back_coords[np.random.randint(len(back_coords))]
            seed_2 = back_coords[np.random.randint(len(back_coords))]
            accepted_range = np.arange(10, 20)
            while (
                i_dist_between_seeds(seed_1, seed_2) not in accepted_range
                or j_dist_between_seeds(seed_1, seed_2) not in accepted_range
            ):
                seed_1 = back_coords[np.random.randint(len(back_coords))]
                seed_2 = back_coords[np.random.randint(len(back_coords))]
            unconnected_seed_pairs.append(sorted([seed_1, seed_2]))
    return unconnected_seed_pairs


def draw_twig_veins(base_veins):
    """
    Add twig veins based on the base veins. The proposed algorithm is as follows:
            - Twig veins propagate out of existing veins towards another random seed point not required to be
                    within the current vein structure.
            - They follow the same propagation algorithm as generating the base veins.
            - Dilate twig veins by a random acceptable factor (typically thinner than main veins)

    Parameters:
    -----------
    base_veins: numpy.array:
            Image from which to determine twig vein seed point pairs.

    Returns:
    --------
    numpy.array
            Black image with added twig veins.
    """
    im = initialize_im()
    twig_point_pairs = generate_twig_vein_seed_pairs(base_veins)
    if len(twig_point_pairs) > 0:
        im = propagate_and_draw_veins(im, twig_point_pairs)
        im = dilate(im, 1.5)
    return im


def draw_unconnected_veins(base_veins):
    """
    Add unconnected veins based on the current vein information. The proposed algorithm is as follows:
            - Unconnected veins are not required to be connected to the existing vein structure (simulate acquisition noise).
            - They follow the same propagation algorithm as generating the base veins.
            - Dilate unconnected veins by an acceptable factor (typically thinner than base veins).

    Parameters:
    -----------
    base_veins: numpy.array:
            Image from which to determine unconnected vein seed point pairs.

    Returns:
    --------
    numpy.array
            Black image with added unconnected veins.
    """
    im = initialize_im()
    unconnected_seed_pairs = generate_unconnected_vein_seed_pairs(base_veins)
    if len(unconnected_seed_pairs) > 0:
        im = propagate_and_draw_veins(im, unconnected_seed_pairs)
        im = dilate(im, 1.5)
    return im


def union_vein_ims(base_veins, twig_veins, unconnected_veins):
    """
    Union the generated trunk and branch hand vein images.

    Parameters:
    -----------
    base_veins: numpy.array
            Base vein image.

    twig_veins: numpy.array
            Twig vein image.

    unconnected_veins: numpy.array
            Unconnected vein image.
    Returns:
    --------
    numpy.array
            Final vein image.
    """
    final_veins = initialize_im()
    for i in range(final_veins.shape[0]):
        for j in range(final_veins.shape[1]):
            if (
                twig_veins[i, j] > 0
                or unconnected_veins[i, j] > 0
                or base_veins[i, j] > 0
            ):
                final_veins[i, j] = 1
    return final_veins


def apply_spatial_variation(final_veins):
    """
    Apply spatial variation to final veins. The proposed algorithm is as follows:
            - Randomly translate final_veins in acceptable range.
            - Randomly rotate translated final_veins in acceptable range.
    The purpose of this is to simulate the variation in orientation that may occur
    while acquiring multiple hand vein samples of actual individuals.

    Parameters:
    -----------
    final_veins: numpy.array
            Final vein image.

    Returns:
    --------
    numpy.array
            Translated and rotated final_veins.
    """
    angle = np.random.uniform(-1, 1)
    tx = np.random.uniform(-1, 1)
    ty = np.random.uniform(-1, 1)
    transformation_matrix = np.float32([[1, 0, tx], [0, 1, ty]])
    final_veins = cv2.warpAffine(
        final_veins,
        transformation_matrix,
        (final_veins.shape[1], final_veins.shape[0]),
        cv2.BORDER_WRAP,
    )
    return tf.rotate(final_veins, angle, mode="wrap")


def main_function(root_output_dir, base_veins, ind, num_ims):
    """
    Orchestrator function for simulating multiple hand vein acquisitions from a given base vein image.
    Algorithm is defined as follows:
            - Draw twig veins.
            - Draw unconnected vein-like structures to the image (simulate segmentation noise).
            - Union the base veins, twig veins and unconnected veins.
            - Repeat num_ims times in order to acquire num_ims simulated acquired hand vein images.

    Parameters:
    -----------
    root_output_dir: str
            Root output directory of results.

    base_veins: numpy.array
            Base vein image.

    ind: int
            Numbered fictitious individual (for naming output files).

    num_ims: int
            Number of images to generate for each fictitious individual.

    Returns:
    --------
    list
            Binary artificial hand vein images for fictitious individual # ind.
    """
    sims = []
    for num_sims in range(num_ims):
        twig_veins = draw_twig_veins(base_veins)
        branch_veins = draw_unconnected_veins(base_veins)
        final_veins = union_vein_ims(base_veins, twig_veins, branch_veins)
        final_veins = apply_spatial_variation(final_veins)
        sims.append(final_veins)
        save(root_output_dir, final_veins, ind, num_sims, "Final_Veins")
    return sims
