import numpy as np
import random


def run_trials(n_trials, dim, count_cumdist, psf_cum):
    """
    Uses a random number generator to sample images produced on a CCD by a single source (ion or background), based on
    the point spread function and emission distribution of the source.

    Parameters
    ----------
    n_trials : int
    dim : tuple
        Dimensions of the CCD
    count_cumdist : array-like
    psf_cum : array-like
        Cumulative flattened point spread function.

        e.g. If for 3 pixels, the PSF is [0.7, 0.2, 0.1], then psf_cum is [0.7, 0.9, 1.0]

    Returns
    -------
    list
        One image for each trial

    """
    counts = []
    for trial in range(n_trials):
        counts.append(np.zeros(np.size(psf_cum)))
        # Determine the number of photons to be emitted from this source based on the cumulative distribution
        rnum = random.random()
        for i in range(np.size(count_cumdist)):
            n_photons = i
            if rnum <= count_cumdist[i]:
                break
        # Randomly place n_photons in 2D array of pixels according to the PSF
        for i in range(0, n_photons):
            rnum = random.random()
            for j in range(np.size(psf_cum)):
                pixel = j
                if rnum <= psf_cum[j]:
                    break
            counts[-1][pixel] += 1
        counts[-1].reshape(dim)
    return counts
