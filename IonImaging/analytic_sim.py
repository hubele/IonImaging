import misc
import numpy as np
from IonImaging import gamma, delta, Delta_1, Delta_2
from scipy.special import comb, factorial
from math import factorial

from scipy.special._ufuncs import gammainc

from numba import int64, float64, NumbaWarning
import warnings

warnings.simplefilter('ignore', category=NumbaWarning)

from . import misc


def get_countdist(pixels, count_dist, psf, max_n=None):
    """
    Calculates the distribution of photon counts for given pixel(s) from a single source (background or ion).

    Parameters
    ----------
    pixels : int or list
    count_dist : array-like
    psf : array-like
        1D
    max_n : int (optional)
        Maximum number of photon counts to consider

    Returns
    -------
    ndarray
        Distribution of photon counts for given pixel(s).
    """

    if max_n is None:
        max_n = np.size(count_dist)

    pixels = np.array([pixels]).flatten()
    n_pixels = np.size(pixels)
    fact_array = np.array([factorial(i) for i in range(max_n)])
    beta = []
    for pixel in pixels:
        beta.append((psf[pixel] ** np.arange(max_n)) / fact_array)
    beta = np.meshgrid(*beta)
    beta = np.prod(beta, axis=0)
    shape = np.shape(beta)

    psf_sum = np.sum([psf[pixel] for pixel in pixels])
    alpha = np.zeros(np.size(count_dist))
    for n in range(np.size(count_dist)):
        for k in range(np.size(count_dist) - n):
            alpha[n] += count_dist[n + k] * ((1 - psf_sum) ** (k)) * factorial(n + k) / factorial(k)

    m_tot = [np.arange(shape[i]) for i in range(np.size(shape))]
    m_tot = np.sum(np.meshgrid(*m_tot), axis=0)
    alpha_array = misc.array_by_indices(m_tot, alpha)

    p_dist = alpha_array * beta
    if n_pixels > 1:
        p_dist = np.swapaxes(p_dist, 0, 1)
    return p_dist.astype(float)


def calc_dists(tau, s, na, qe, r_bg, lambda_0=None):
    """
    Calculates the collection distributions (probability of measuring certain number of photons on entire CCD) from a
    bright ion, dark ion, and background.

    Parameters
    ----------
    tau : float
        Detection time (microseconds)
    s : float
        I/I_sat
    na : float
        Numerical aperture
    qe : float
        Quantum efficiency
    r_bg : float
        Background rate over entire CCD (counts/ms)
    lambda_0 : float (optional)
        If specified, sets the average photon count from an ion which is bright over the entire duration of measurement

    Returns
    -------
    ndarray
        Bright state distribution
    ndarray
        Dark state distribution
    ndarray
        Background distribution

    """
    eta = qe * misc.efficiency(na)  # total collection efficiency
    tau = tau * 1e-6

    if lambda_0 is None:
        lambda_0 = tau * eta * s * (gamma / 2) / (1 + s + (2 * delta / gamma) ** 2)

    alpha_1 = (2 / 9) * (1 + s + (2 * delta / gamma) ** 2) * (gamma / 2 / Delta_1) ** 2
    alpha_2 = (2 / 9) * (1 + s + (2 * delta / gamma) ** 2) * (gamma / 2 / Delta_2) ** 2

    # The 2/9 branching ratio is derived with the following calculation. Here we assume all three polarizations are
    # equally strong. (https://qiti-serv.iqc.uwaterloo.ca/QITI/Data/2020/01/30/branching%20ratio.png)

    alpha_1_over_eta = alpha_1 / eta
    alpha_2_over_eta = alpha_2 / eta
    n = np.arange(50)  # number of counts

    # Theoretical probability distribution of photon number of dark state
    # Average photon counts from the ion
    p_dark = np.exp(-alpha_1_over_eta * lambda_0) * (alpha_1_over_eta / (1 - alpha_1_over_eta) ** (n + 1) *
                                                     gammainc(n + 1, (1 - alpha_1_over_eta) * lambda_0) + (n == 0))

    # Theoretical probability distribution of photon number of bright state
    # Average photon counts from the ion
    p_bright = np.exp(-(1 + alpha_2_over_eta) * lambda_0) * lambda_0 ** n / factorial(n) + alpha_2_over_eta / (
            1 + alpha_2_over_eta) ** (n + 1) * gammainc(n + 1, (1 + alpha_2_over_eta) * lambda_0)

    # Poisson noise distribution
    lambda_bg = tau * r_bg / 1e-3  # Average photon counts from noise
    p_bg = np.exp(-lambda_bg) * lambda_bg ** n / factorial(n)

    return p_bright, p_dark, p_bg
