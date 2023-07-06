# -*- coding: utf-8 -*-
"""
Created on Sat May 16 14:14:07 2020

@author: Scott Hubele
"""
import warnings

import numpy as np
import scipy.special as special
from numba import int64, float64, NumbaWarning

warnings.simplefilter('ignore', category=NumbaWarning)

from . import _judge_state
from . import misc
from .analytic_sim import *
from .classes import *
from .montecarlo_sim import *


class NumpySimulation(Simulation):
    """
    Simulation using NumPy.

    Simulate photon counts using NumPy for Monte-Carlo or analytic expressions. Methods exist to calculate the
    measurement fidelity for a certain method of judging ion states, plot average counts in each pixel, and plots the
    point spread function of all photon sources. State judgement can be performed on experiments with 1 ion captured
    by 1 or 2 pixels.

    Attributes
    ----------
    dim : list of int
    sources : list of Ion or Bkg objects or None
    tau : float or None
    s : float or None
    na : float or None
    qe : float or None
    """

    def __init__(self, dim, tau, s, na, qe, r_bg, sources=None):
        """

        Parameters
        ----------
        dim : list of int
            n_y pixels, n_x pixels
        tau : float
            Detection time (microseconds)
        s : float
            Saturation parameter
        na : float
            Numerical aperture
        qe : float
            Quantum efficiency
        r_bg : float
            Average background rate (counts/ms/pixel)
        sources : list (optional)
            Ion or Bkg objects
        """
        super().__init__(dim, tau, s, na, qe, r_bg, sources)

        self.mc_counts = []
        self.anal_dists = []

    def mc_sim(self, n_trials=0):
        """
        Runs Monte-Carlo simulations to determine number of photons counted by each pixel for each run.

        Parameters
        ----------
        n_trials : int
            Number of trials

        Returns
        -------
        list of ndarray objects
            Each array in list gives photon counts in each pixel for a particular trial.

        """
        if n_trials != 0:
            self.mc_counts = []
        for src in self.sources:
            count_cumdist = np.cumsum(src.count_dist)
            psf_cum = np.cumsum(src.psf)
            if n_trials != 0:
                self.mc_counts.append(run_trials(n_trials, self.dim, count_cumdist, psf_cum))
        return np.sum(self.mc_counts, axis=0)  # sum all sources together

    def anal_sim(self, pixels):
        """
        Calculates the distribution of photon counts for given pixel(s) for all sources in self.sources.

        Parameters
        ----------
        pixels : int or list

        Returns
        -------
        ndarray
            Distribution of photon counts. Number of dimensions matches number of given pixels. Element (n1,n2,...) is
            the probability of counting n1 photons in first given pixel, n2 photons in second given pixel, etc.

        """
        self.anal_dists = []
        for src in self.sources:
            self.anal_dists.append(get_countdist(pixels, src.count_dist, src.psf))
        return misc.convolve(*self.anal_dists)

    def plot_averages(self, method, shape, **kwargs):
        """
        Computes and plots average photon counts, based on a Monte-Carlo or analytic simulation.

        Parameters
        ----------
        method : str
            'mc' or 'analytic'
        shape : tuple
            nrows, ncols of subplots
        kwargs:
            cbar_label : str
            data : list of 2d arrays
            exp : Experiment object
            shape : tuple
            show_vals : bool
            sigdigs : int
            hide_axes : bool
            plot_cbar : bool
            cmap : str

        Returns
        -------

        """
        if method == 'mc':
            n_trials = 0
            if 'n_trials' in kwargs:
                n_trials = kwargs.pop('n_trials')
            self.mc_sim(n_trials)
            averages = np.average(self.mc_counts, axis=1)
        elif method == 'analytic':
            averages = []
            for src in self.sources:
                averages.append([])
                for i in range(self.dim[0] * self.dim[1]):
                    dist_pixel_i = get_countdist(i, src.count_dist, src.psf)
                    averages[-1].append(np.sum(dist_pixel_i * np.arange(dist_pixel_i.size)))
            averages = np.array(averages)

        if not 'cbar_label' in kwargs:
            kwargs['cbar_label'] = 'Average Counts'
        visualize.plot_imgs(averages, self, shape, **kwargs)

    def measure_fidelity(self, pixels, method='max_post', plot=True, **kwargs):
        """
        Prints the bright and dark state fidelity based on a given method of judging state based on the photon counts
        in two pixels. Currently only works when there is only one ion in the experiment.

        Parameters
        ----------
        pixels : list of int
        method : 'max_post' | 'compp_rand' | thresh_5050' | 'thresh_diag' | 'thresh_1d'

            'max_post' :    1 ion, 2 detectors. Maximum a posteriori, assigns a state based on which state has the
                            highest probability of producing the measured set of photon counts.

            'compp_rand' :  1 ion, 2 detectors. Randomly assigns a state, biasing results towards the state with the
                            higher probability of producing the measured set of photon counts.

            'thresh_5050' : 1 ion, 2 detectors. Performs a threshold judgement on each detector separately. If detectors
                            agree, assign the state to match. If the detectors disagree, assign the state of the ion
                            randomly with no bias.

            'thresh_diag' : 1 ion, 2 detectors. Performs a threshold judgement on a weighted sum of the counts in each
                            detector.

            'thresh_1d'   : 1 ion, 1 detector.

        plot : bool
        kwargs : dict
            Used with threshold methods, for setting threshold and/or weights. If thresholds and weights are not
            specified they will be calculated to maximize the average fidelity.

        Returns
        ----------
        list of floats
            Fidelity values for each possible state
        """

        plot_1d = False
        initial_state = self.get_state()
        states = misc.decimal_to_binary(*np.arange(2 ** (len(self.ions))))
        dists = []
        for state in states:
            dists.append([])
            self.set_state(state)
            for src in self.sources:
                dists[-1].append(get_countdist(pixels, src.count_dist, src.psf))
            dists[-1] = misc.convolve(*dists[-1])
        self.set_state(initial_state)  # so that the state of the system is not modified by calling this function

        if method == 'compp_disc':
            measured_states = self._max_post(dists)
            f = _judge_state.measure_f_discrete(states, dists, measured_states)
            colours = _judge_state.assign_colours_discrete(measured_states)
        elif method == 'compp_rand':
            states_probs = self._compp_rand(dists)
            f = _judge_state.measure_f_rand(dists, states_probs)
            colours = _judge_state.assign_colours_rand(states_probs)
        elif method == 'thresh_5050':
            if 'threshold' in kwargs:
                threshold = kwargs.pop('threshold')
                states_probs = self._thresh_5050(dists[0].shape, threshold)
                f = _judge_state.measure_f_rand(dists, states_probs)
            else:  # Threshold not given - calculate threshold for maximum f
                f_max = 0
                thresh_ideal = [0, 0]
                for thresh_1 in range(0, 20):
                    for thresh_2 in range(0, 20):
                        states_probs = self._thresh_5050(dists[0].shape, threshold=[thresh_1, thresh_2])
                        f = _judge_state.measure_f_rand(dists, states_probs)
                        if np.average(f) > f_max:
                            f_max = np.average(f)
                            thresh_ideal = [thresh_1, thresh_2]
                states_probs = self._thresh_5050(dists[0].shape, thresh_ideal)
                f = _judge_state.measure_f_rand(dists, states_probs)

            colours = _judge_state.assign_colours_rand(states_probs)
        elif method == 'thresh_diag':
            if 'weights' in kwargs:
                weights = kwargs.pop('weights')
            else:
                raise SyntaxError('Weights not specified.')
            if 'threshold' in kwargs:
                threshold = kwargs.pop('threshold')
                states_probs = self._thresh_diag(dists[0].shape, weights, threshold)
                f = _judge_state.measure_f_discrete(['0', '1'], dists, states_probs)
            else:  # Threshold not given - calculate threshold for maximum f
                f_max = 0
                thresh_ideal = [0, 0]
                for thresh in range(0, 20):
                    states_probs = self._thresh_diag(dists[0].shape, weights, thresh)
                    f = _judge_state.measure_f_discrete(['0', '1'], dists, states_probs)
                    if np.average(f) > f_max:
                        f_max = np.average(f)
                        thresh_ideal = thresh
                states_probs = self._thresh_diag(dists[0].shape, weights, thresh_ideal)
                f = _judge_state.measure_f_discrete(['0', '1'], dists, states_probs)
            colours = _judge_state.assign_colours_discrete(states_probs)
        elif method == 'thresh_1d':
            if 'threshold' in kwargs:
                threshold = kwargs.pop('threshold')
                states_probs = self._thresh_1d(dists[0].size, threshold)
                f = _judge_state.measure_f_discrete(['0', '1'], dists, states_probs)
            else:  # Threshold not given - calculate threshold for maximum f
                f_max = 0
                thresh_ideal = 0
                for thresh in range(0, 20):
                    states_probs = self._thresh_1d(dists[0].size, thresh)
                    f = _judge_state.measure_f_discrete(['0', '1'], dists, states_probs)
                    if np.average(f) > f_max:
                        f_max = np.average(f)
                        thresh_ideal = thresh
                states_probs = self._thresh_1d(dists[0].size, thresh_ideal)
                f = _judge_state.measure_f_discrete(['0', '1'], dists, states_probs)
                kwargs['threshold'] = thresh_ideal
                kwargs['labels'] = ['0', '1']
                if plot:
                    plot = False
                    plot_1d = True

        else:
            raise ValueError("Invalid method given.")

        if plot:
            self._plot_2d_dist(dists, colours, **kwargs)
        if plot_1d:
            visualize.bar_1d(dists, **kwargs)

        return f

    def _max_post(self, dists):
        """
        Judges the state of a single ion based on the photon counts n1 and n2 in detectors 1 and 2 for each set of
        (n1, n2), according to maximum a posteriori (a measurement (n1,n2) is judged as the state with the highest
        probability P(n1,n2)).

        Parameters
        ----------
        dists : list of ndarrays
            0 : Probability of measuring photon counts (n1,n2) from a dark ion
            1 : Probability of measuring photon counts (n1,n2) from a bright ion

        Returns
        -------
        ndarray
            S(n1,n2), the measured state for each set of photon counts (n1,n2).

        """

        n_states = len(dists)
        states = misc.decimal_to_binary(*np.arange(n_states))
        dtype = '<U' + str(len(self.ions))
        probable_states = np.empty(shape=dists[0].shape, dtype=dtype)

        # For each n1,n2 find state with maximum probability
        for n1 in range(dists[0].shape[0]):
            for n2 in range(dists[0].shape[1]):
                p_n1_n2 = [dists[i][n1, n2] for i in range(len(dists))]
                probable_states[n1, n2] = states[np.argmax(p_n1_n2)]

        return probable_states

    def _compp_rand(self, dists):
        """
        Judges the state of a single ion based on the photon counts n1 and n2 in detectors 1 and 2 for each set of
        (n1, n2), using a random number generator. For a set of photon counts (n1,n2) the state is randomly determined
        where the probability of judging the ion as bright is equal to P_b(n1,n2)/[P_b(n1,n2)+P_d(n1,n2)].

        Parameters
        ----------
        dists : list of ndarrays
            0 : Probability of measuring photon counts (n1,n2) from a dark ion, P_d(n1,n2)
            1 : Probability of measuring photon counts (n1,n2) from a bright ion, P_b(n1,n2)

        Returns
        -------
        list of ndarray
            0: S_0(n1,n2), the probability of measuring state 0, when photon counts (n1,n2) are recorded
            1: S_1(n1,n2), the probability of measuring state 1, when photon counts (n1,n2) are recorded

        """

        n_states = len(dists)
        p_sum = np.sum(dists, axis=0)

        state_probs = []
        for i in range(n_states):
            state_probs.append(dists[i] / p_sum)

        return state_probs

    def _thresh_5050(self, max_n, threshold):
        """
        Used for a single ion imaged by two detectors. Uses 1D threshold method on pixel counts for both pixels
        separately and assigns a state only if both detectors agree. If detectors do not agree, assigns state randomly
        with 50/50 biasing.

        Parameters
        ----------
        max_n : list
            Maximum number of photon counts to consider in each detector.
        threshold : list
            0 : threshold with which photon count n1 is compared
            1 : threshold with which photon count n2 is compared


        Returns
        -------
        list of ndarray
            0: S_0(n1,n2), the probability of measuring state 0, when photon counts (n1,n2) are recorded
            1: S_1(n1,n2), the probability of measuring state 1, when photon counts (n1,n2) are recorded
        """

        # 2D implementation

        dark_prob = np.zeros(max_n, dtype=float)
        bright_prob = np.zeros(max_n, dtype=float)
        for n1 in range(max_n[0]):
            for n2 in range(max_n[1]):
                if n1 <= threshold[0]:
                    if n2 <= threshold[0]:
                        dark_prob[n1, n2] = 1.0
                        bright_prob[n1, n2] = 0.0
                    else:
                        dark_prob[n1, n2] = 0.5
                        bright_prob[n1, n2] = 0.5
                else:
                    if n2 <= threshold[0]:
                        dark_prob[n1, n2] = 0.5
                        bright_prob[n1, n2] = 0.5
                    else:
                        dark_prob[n1, n2] = 0.0
                        bright_prob[n1, n2] = 1.0
        states_probs = [dark_prob, bright_prob]
        return states_probs

        # nd implementation
        # n_states = 2 ** (len(dists))
        # states = misc.decimal_to_binary(*np.arange(n_states))
        # dtype = '<U' + str(len(self.ions))
        # probable_states = np.empty(shape=(dists[0].shape), dtype=dtype)
        #
        # pixel_dists = []
        # for dist in dists:
        #     pixel_dists.append([])
        #     for i in range(len(dist.shape)):
        #         sum_axes = list(np.arange(len(dist.shape)))
        #         sum_axes.pop(i)
        #         sum_axes = tuple(sum_axes)
        #         pixel_dists[-1].append(np.sum(dist, axis=sum_axes))
        # pixel_dists = np.array(pixel_dists)

    def _thresh_diag(self, shape, weights, threshold):
        """
        Used for a single ion imaged by two detectors. Performs a threshold judgement on the weighted sum of the photon
        counts in the two detectors.

        Parameters
        ----------
        ndarray
            S(n1,n2), the measured state for each set of photon counts (n1,n2).
        """
        probable_states = np.empty(shape, dtype='<U1')

        for n1 in range(shape[0]):
            for n2 in range(shape[1]):
                n_weighted = weights[0] * n1 + weights[1] * n2
                probable_states[n1, n2] = str(int(n_weighted > threshold))

        return probable_states

    def _thresh_1d(self, n_max, threshold):
        """

        Parameters
        ----------
        n_max : int
            Maximum number of photon counts to consider.
        threshold : int

        Returns
        -------
        ndarray
            S(n), the measured state for each possible number of photon counts in a single detector.
        """
        probable_states = np.empty(n_max, dtype='<U1')

        for n in range(n_max):
            probable_states[n] = str(int(n > threshold))

        return probable_states

    def _plot_2d_dist(self, dists, colours=None, n1_max=30, n2_max=30, auto_colour=True):
        """
        Plots the 2D probability distribution for photon counts in two given pixels.

        Parameters
        ----------
        n1_max : int
        n2_max : int
        states : '00' | '01' | '100' | etc.
            Number of digits must match number of ions.
        auto_colour : bool

        Returns
        -------

        """
        states = misc.decimal_to_binary(*np.arange(2 ** (len(self.ions))))
        if colours is not None:
            colours = colours[:n1_max, :n2_max]
            colours = colours.flatten()
        visualize.bar_2d(dists, n1_max, n2_max, colours, labels=states)
