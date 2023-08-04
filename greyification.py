# Third party imports
import numpy as np
from skimage.filters import median, rank, gaussian
from skimage.morphology import binary_erosion, erosion, disk

# Local imports
from utils import initialize_im, save


def make_greyscale(final_veins):
    """
    Initial greyification of a binary hand vein image. The proposed algorithm is outlined as follows:
            - Initialize a black image of same shape as final_veins.
            - Loop through final_veins and determine whether current pixel is a vein or background pixel.
            - Randomly choose greyscale vein or background intensity for current pixel from acceptable intensity ranges.
            - Add minor variation to choice.
            - Set corresponding pixel in new image equal to newly obtained intensity.

    Parameters:
    -----------
    final_veins: numpy.array
            Thinned version of a given binary hand vein image.

    Returns:
    --------
    numpy.array
            Initially greyified version of final_veins.
    """
    new_im = initialize_im()
    for i in range(final_veins.shape[0]):
        for j in range(final_veins.shape[1]):
            var = np.random.uniform(-0.05, 0.05)
            if final_veins[i, j] > 0:
                vein_intensity = np.random.uniform(0.2, 0.3)
                new_im[i, j] = vein_intensity + var
            else:
                background_intensity = np.random.uniform(0.05, 0.15)
                new_im[i, j] = background_intensity + var
    return new_im


def main_function(root_output_dir, struct_veins, ind):
    """
    Orchestrator function for creating a grey-scale version of the artificial binary hand vein images.
    The proposed algorithm is outlined as follows:
            - First apply minor erosion of the hand vein image, since successive smoothing will thicken vein structures.
            - Randomly greyify vein and background pixels by selecting from an acceptable intensity range.
            - Apply successive smoothing with local mean and median filters.
            - Apply intermittent grey-scale erosion in order to avoid blob-like structures.

    Parameters:
    -----------
    root_output_dir: str
            Output directory to save results.

    struct_veins: list
            List of final_veins images for given fictitious individual.

    ind: int
            Numbered fictitious individual.
    """
    for i in range(len(struct_veins)):
        f_vein = struct_veins[i]
        img = binary_erosion(f_vein, selem=disk(3))
        img = make_greyscale(img)
        img = rank.mean(img, selem=disk(2))
        img = median(img, disk(2))
        img = rank.mean(img, selem=disk(1.5))
        img = median(img, disk(1.5))
        img = rank.mean(img, selem=disk(1.5))
        img = median(img, disk(1.5))
        img = gaussian(img, 2.5)
        img = erosion(img, selem=disk(1.5))
        save(root_output_dir, img, ind, i, "Original")
