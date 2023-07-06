# -*- coding: utf-8 -*-
"""
Created on Sat May 16 14:14:07 2020

@author: Scott Hubele
"""

from . import misc
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

from scipy import special
from scipy.special._ufuncs import gammainc

from numba import NumbaWarning
import warnings

warnings.simplefilter('ignore', category=NumbaWarning)

# DATA (Properties of 171Yb+ and laser detuning)
gamma = 20e6  # Natural linewidth
delta = 0  # Detuning from transition
Delta_1 = 14.7e9  # HFS + HFP
Delta_2 = 2.1e9  # HFP


# Classes

class PhotonSource:
    """
    Base class for Ion and Bkg classes.

    Attributes
    ----------
    count_dist : array (optional)
    psf : array-like
    label : str

    """

    def __init__(self, count_dist=None, psf=None, label=''):
        self.label = label
        self.count_dist = count_dist
        self.psf = np.array(psf).flatten()


class Ion(PhotonSource):
    """
    Class to represent trapped Yb+ ion. Should be created by Experiment.add_ion.

    """

    def __init__(self, *args, state=1, pos=None, **kwargs):
        """

        Parameters
        ----------
        args
            Passed to PhotonSource init
        state : int
            0 or 1
        pos : list
            x_pos, y_pos
        kwargs
            Passed to PhotonSource init
        """
        super().__init__(*args, **kwargs)
        self.state = bool(state)
        self.pos = pos


class Bkg(PhotonSource):
    """
    Class to represent any/all background sources in a run. Uses a Poisson noise distribution if a count distribution is
    not specified.

    """

    def __init__(self, count_dist=None, psf=None, label=''):
        """

        Parameters
        ----------
        count_dist : ndarray (optional)
        psf : array-like
        label : str
        """

        super().__init__(count_dist, psf, label)


class Simulation:
    """
    Base class for NumpySimulation and TFPSimulation.

    Contains CCD dimension, experimental parameters, photon sources (ions/background), and experimental parameters.
    Methods exist to add ions/background to the run. Further methods exist to determine and set the state of the ions,
    and to set the position of ions on the CCD.

    Functions do not check whether attribute are none or not, pass in as much information as possible, and do not call
    methods that require variables that haven't been defined.

    Attributes
    ----------
    dim : list of int
        n_y pixels, n_x pixels
    sources : list
        Ion or Bkg objects
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
    spot_diam : float
        Simulated size of ion spot on CCD (in pixels)
    ccd_pos : list
        x_min, x_max, y_min, y_max
    """

    def __init__(self,
                 dim=None,
                 tau=None,
                 s=None,
                 na=None,
                 qe=None,
                 r_bg=None,
                 spot_diam=None,
                 spacing=None,
                 angle=None,
                 ccd_pos=None,
                 sources=[]
                 ):
        """

        Parameters
        ----------
        dim : list
            Shape of CCD (number of pixels) (y,x)
        tau : float
            Detection time (microseconds)
        s : float
            Saturation parameter, I/I_sat
        na : float
            Numerical aperture
        qe : float
            Quantum efficiency of each pixel (same for each)
        r_bg : float
            Average background rate (counts/ms/pixel)
        spot_diam : float
            Diameter of ion spot on CCD, measured in pixels
        spacing : float
            Nominal centre-to-centre spacing of ions, measured in pixels. The spacing is the same between all ions in
             the same run, but spacing will differ run to run by ~20%.
        angle : float
            Angle of ion chain relative to x-axis
        ccd_pos : list (optional)
            Assigns values to the position of each edge of the CCD (x_min, x_max, y_min, y_max)
        sources : list of PhotonSource objects (optional)

        """

        if ccd_pos is None:
            if dim is not None:
                ccd_pos = [0, dim[1], 0, dim[0]]

        self.tau = tau  # Detection time (microseconds)
        self.s = s  # Saturation parameter I/I_Sat
        self.na = na  # Numerical aperture
        self.qe = qe  # Quantum efficiency of CCD
        self.eta = qe * misc.efficiency(na)  # Total collection efficiency
        self.r_bg = r_bg  # Average background rate (per pixel)
        self.spot_diam = spot_diam
        self.spacing = spacing
        self.angle = angle
        self.ccd_pos = ccd_pos
        self.lambda_0 = self.get_lambda_0()

        self.dim = dim
        self.sources = []
        self.ions = []
        self.bkg = []
        self.src_groups = []
        self._sort_sources(sources)

        self.psf_ions = self.get_psf_ions()
        self.psf_bkg = self.get_psf_bkg()

        self.bright_dist, self.dark_dist, self.bkg_dist = self.get_pdists()

    def set_state(self, state):
        """
        Sets the state of all ions in the experiment.

        Parameters
        ----------
        state : str
            i.e. '10' | '010' | etc.
            Number of digits in given string must match the number of ions.

        Returns
        -------

        """
        if len(state) != len(self.ions):
            raise ValueError('Given state does not match number of ions')
        for i in range(len(state)):
            single_ion_state = int(state[i])
            self.ions[i].state = single_ion_state
            if bool(single_ion_state) is True:
                self.ions[i].count_dist = self.bright_dist
            elif bool(single_ion_state) is False:
                self.ions[i].count_dist = self.dark_dist

    def get_state(self, fmt='bin_str', **kwargs):
        """
        Returns the current state of the ions.

        fmt : 'bin_str' or 'bin_array'
        kwargs :
            Correspond to given fmt, see misc module

        Returns
        -------
        str

        """
        state = [int(ion.state) for ion in self.ions]
        state = [np.array(state)]
        state, = misc.get_state(state, fmt, **kwargs)
        return state

    def add_ion(self, psf=None, label='', state=None, count_dist=None):
        """
        Creates and returns Ion object, adding it to the list of sources in the experiment. If state is specified, a
        given count distribution will be overridden by the calculated distribution from the experimental parameters.

        Parameters
        ----------
        psf : list (optional)
        label : str (optional)
        state : int (optional)
            0 or 1
        count_dist : list (optional)

        Returns
        -------
        Ion object

        """

        if psf is not None:
            if np.size(psf) != np.prod(self.dim):
                raise ValueError("Size of PSF must match number of pixels")
        if count_dist is None:
            count_dist = self.bright_dist
        ion = Ion(count_dist, psf, label, state=state)
        if state is not None:
            if bool(state) is True:
                ion.count_dist = self.bright_dist
            elif bool(state) is False:
                ion.count_dist = self.dark_dist
        self.ions.append(ion)
        self.sources.append(ion)

        self.psf_ions = self.get_psf_ions()

        return ion

    def add_bkg(self, count_dist=None, psf=None, label=''):
        """
        Creates and returns Bkg object, adding it to the list of sources in the experiment. If a count distribution
        is not specified it will be calculated when added to an experiment based on the experimental parameters.

        Parameters
        ----------
        count_dist : list
        psf : list
        label : str

        Returns
        -------
        Bkg object

        """
        if psf is None:
            psf = np.ones(np.prod(self.dim)) / (np.prod(self.dim))
        if count_dist is None:
            count_dist = self.bkg_dist
        bkg = Bkg(count_dist, psf, label)
        self.bkg.append(bkg)
        self.sources.append(bkg)

        self.psf_bkg = self.get_psf_bkg()

        return bkg

    def get_psf_ions(self):
        """
        Returns the point spread function of all ions in the run.

        Returns
        -------
        list
            PSF of each ion

        """
        psf_ions = []
        for ion in self.ions:
            if ion.psf is None:
                psf_ions.append(None)
            else:
                psf_ions.append(ion.psf.astype('float32'))

        return psf_ions

    def set_psf_ions(self, psf_list):
        """
        Sets the PSF of the first X ions in the simulation.

        Parameters
        ----------
        psf_list : list
            Values are of type ndarray or None

        Returns
        -------

        """
        for i in range(len(psf_list)):
            self.ions[i].psf = psf_list[i]

    def get_psf_bkg(self):
        """
        Returns the PSF of all background sources in the experiment.

        Returns
        -------
        list
            PSF of each background source.

        """
        psf_bkg = []
        for bkg in self.bkg:
            psf_bkg.append(bkg.psf.astype('float32'))
        return psf_bkg

    def plot_psfs(self):
        """
        Plots the point-spread function of all photon sources in the experiment.

        Returns
        -------

        """
        psf_list = []
        labels = []
        for src in self.sources:
            if src.psf is not None:
                psf_list.append(src.psf.reshape(self.dim).tolist())
                labels.append(src.label)

        lims = [np.min(psf_list), np.max(psf_list)]
        fig, ax = plt.subplots()
        plt.subplots_adjust(right=0.80, bottom=0.20)
        i_src = 0
        plt.title(labels[0])
        im = ax.matshow(psf_list[i_src], cmap='gray', vmin=lims[0], vmax=lims[1])  # , aspect='auto')
        ax.set_xticks([])
        ax.set_yticks([])

        cax = plt.axes([0.85, 0.2, 0.05, 0.7])
        fig.colorbar(im, cax=cax, label='PSF')

        axcolor = 'lightgoldenrodyellow'
        ax_src = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)

        slider = Slider(ax_src, 'Source', 0, len(psf_list) - 1, valinit=i_src, valstep=1)

        def update(val):
            i_src = int(slider.val)
            im.set_data(psf_list[i_src])
            ax.set_title(labels[i_src])
            fig.canvas.draw_idle()

        slider.on_changed(update)

        plt.show()

    def get_lambda_0(self):
        """
        Calculates lambda_0, the average number of photons collected on the CCD from a bright state ion assuming the ion
        does not decay, based on parameters in the experiment.

        Returns
        -------

        """
        if None in [self.tau, self.s, self.eta]:
            return None
        else:
            return self.tau * 1e-6 * self.eta * self.s * (gamma / 2) / (1 + self.s + (2 * delta / gamma) ** 2)

    def list_psfs(self, src_type='all'):
        """
        Returns the PSF of specified sources.

        Parameters
        ----------
        src_type : 'ions' | 'bkg' | 'all'
            Sources for which to list point-spread functions

        Returns
        -------
        list
            PSF of each source of specified type

        """
        if src_type is 'ions':
            return [src.psf for src in self.ions]
        elif src_type is 'bkg':
            return [src.psf for src in self.bkg]
        elif src_type is 'all':
            return [src.psf for src in self.sources]
        else:
            raise ValueError('Invalid src_type specified')

    def list_ion_pos(self):
        """
        Returns the position on the CCD of each ion in the experiment.

        Returns
        -------
        list
            Positions given as (y,x)

        """
        return [ion.pos for ion in self.ions]

    def randomize_ion_pos(self, n_ions=None):
        """
        Places a chain of ions on the CCD, and sets the positions of each ion in the experiment according to the
        position of the chain, in order from left to right. Ions which are not placed have their position set to None.

        The chain has a random centre location, random angle relative to x-axis, and random (uniform) spacing between
        ions. The spacing of ions is within 20% of the nominal value set in the experimental parameters.

        Parameters
        ----------
        n_ions : int
            Number of ions to use in experiment. If not specified, sets the number of ions randomly, as low as half the
            total number of ions contained in the experiment class.

        Returns
        -------
        list
            Positions of each ion (x,y)

        """
        padding = 2.0  # minimum distance between edge of CCD and ion center
        spacing_w_error = np.random.uniform(0.7 * self.spacing, 1.3 * self.spacing)

        max_imageable_length = np.sqrt(np.sum(np.prod(np.array(self.dim) - 2.0 * padding)))
        max_num_ions = np.floor((max_imageable_length - 0.5) / spacing_w_error)

        max_num_ions = min(len(self.ions), int(max_num_ions))

        if n_ions is None:
            n_ions = np.random.randint(max_num_ions // 2 + max_num_ions % 2, max_num_ions + 1)
        elif n_ions > max_num_ions:
            n_ions = max_num_ions
        angle = np.random.uniform(-90., 90.)

        x_spacing = np.cos(np.pi * angle / 180.) * spacing_w_error
        y_spacing = np.sin(np.pi * angle / 180.) * spacing_w_error

        x_pos = np.arange(n_ions) * x_spacing
        y_pos = np.arange(n_ions) * y_spacing

        x_pos = x_pos - x_pos.min() + padding
        y_pos = y_pos - y_pos.min() + padding

        pos = np.array([x_pos, y_pos]).transpose()

        max_vals = np.max(pos, axis=0)

        free_range = np.array(self.dim).astype(float) - max_vals - padding
        if np.min(free_range) < 0:
            self.randomize_ion_pos(n_ions)
        offset = np.random.uniform(low=0, high=free_range)
        pos = pos + offset
        pos = pos.tolist()

        while len(pos) < len(self.ions):
            pos.append(None)

        self.update_ion_pos(pos)

        return [xy for xy in pos if xy is not None]

    def update_ion_pos(self, pos_list):
        """
        Updates the position of each ion from given list, and updates the psf of each ion.

        Parameters
        ----------
        pos_list : list


        Returns
        -------

        """
        for i in range(len(pos_list)):
            self.ions[i].pos = pos_list[i]
        self.update_psf_from_pos()

    def update_psf_from_pos(self, accuracy=5):
        """
        Sets the PSF of each ion according to their current positions on the CCD. Photon emission from ions is
        simulated as a circular flat-top beam, and the value of the PSF of each pixel is taken as the overlap of then
        circular beam with that pixel.

        Parameters
        ----------
        accuracy : int
            Computation time scales with the square of accuracy

        Returns
        -------
        list

        """

        delta_x = (self.ccd_pos[1] - self.ccd_pos[0]) / float(self.dim[1])
        delta_y = (self.ccd_pos[3] - self.ccd_pos[2]) / float(self.dim[0])

        psf = []
        for i_ion in range(len(self.ions)):
            if self.ions[i_ion].pos is None:
                psf.append(None)
                self.ions[i_ion].psf = None
            else:
                psf.append(np.zeros(self.dim))
                for i in range(self.dim[0]):
                    for j in range(self.dim[0]):
                        xmin = float(self.ccd_pos[0]) + j * delta_x
                        xmax = xmin + delta_x

                        ymin = float(self.ccd_pos[2]) + i * delta_y
                        ymax = ymin + delta_y

                        pts = misc.gen_rect_mesh([accuracy, accuracy], [xmin, xmax, ymin, ymax])

                        n_in_spot = 0
                        for point in pts:
                            if misc.is_in_circle(point, [self.ions[i_ion].pos[0], self.ions[i_ion].pos[1]],
                                                 self.spot_diam / 2.):
                                n_in_spot += 1

                        psf[-1][i, j] = n_in_spot / (accuracy ** 2)

                psf[-1] = psf[-1] / np.sum(psf[-1])
                psf[-1] = psf[-1].astype('float32')
                self.ions[i_ion].psf = psf[-1]

        self.psf_ions = psf

        return psf

    def update_params(self, tau=None, s=None, na=None, qe=None, r_bg=None):
        """
        Updates experimental parameters

        Parameters
        ----------
        tau : float (optional)
            Detection time (microseconds)
        s : float (optional)
            Saturation parameter I/I_sat
        na : float (optional)
            Numerical aperture
        qe : float (optional)
            Quantum efficiency
        r_bg : float (optional)
            Average background rate (counts/ms/pixel)


        Returns
        -------

        """
        if tau is not None:
            self.tau = tau
        if s is not None:
            self.s = s
        if na is not None:
            self.na = na
        if qe is not None:
            self.qe = qe
        if r_bg is not None:
            self.r_bg = r_bg

        self.eta = self.qe * misc.efficiency(self.na)
        self.lambda_0 = self.get_lambda_0()

    def set_ion_psfs(self, psf_list):
        """
        Specify PSF of first X ions.

        Parameters
        ----------
        psf_list : list


        Returns
        -------

        """
        size = len(psf_list)
        for i in range(size):
            self.ions[i].psf = np.array(psf_list[i]).flatten()
        self.psf_ions[:size] = [np.array(psf).flatten() for psf in psf_list]

    def set_bkg_psfs(self, psf_list):
        """
        Specify PSF of first X background sources.

        Parameters
        ----------
        psf_list : list


        Returns
        -------

        """
        size = len(psf_list)
        for i in range(size):
            self.bkg[i].psf = np.array(psf_list[i]).flatten()
        self.psf_bkg[:size] = [np.array(psf).flatten() for psf in psf_list]

    def states_asstr(self):
        """
        Returns the state of all ions in the experiment as a string.

        Returns
        -------
        str
            State of all ions

        """
        states_str = ''.join([str(ion.state) for ion in self.ions])
        return states_str

    def states_asint(self):
        """
        Returns the state of all ions in the experiment as a list of int.

        Returns
        -------
        list of int

        """
        states_int = [int(ion.state) for ion in self.ions]
        return states_int

    def state_asfloat(self):
        """
        Returns the state of all ions in the experiment as a list of float.

        Returns
        -------
        list of float
            State of all ions

        """
        states_float = [float(ion.state) for ion in self.ions]
        return states_float

    def get_pdists(self):
        """
        Calculates the collection distributions (probability of measuring certain number of photons on entire CCD) from a
        bright ion, dark ion, and background.

        Returns
        -------
        p_bright : ndarray
            Bright state distribution
        p_dark : ndarray
            Dark state distribution
        p_bg : ndarray
            Noise distribution

        """

        if None in [self.eta, self.s, self.tau]:
            raise ValueError('Could not calculate distributions due to a None value.')

        if self.lambda_0 is None:
            self.lambda_0 = self.get_lambda_0()

        tau = self.tau * 1e-6

        alpha_1 = (2 / 9) * (1 + self.s + (2 * delta / gamma) ** 2) * (gamma / 2 / Delta_1) ** 2
        alpha_2 = (2 / 9) * (1 + self.s + (2 * delta / gamma) ** 2) * (gamma / 2 / Delta_2) ** 2

        # <!-- The 2/9 branching ratio is derived with the following calculation. Here we assume all three polarizations are equally strong.
        # ![](https://qiti-serv.iqc.uwaterloo.ca/QITI/Data/2020/01/30/branching%20ratio.png) -->

        alpha_1_over_eta = alpha_1 / self.eta
        alpha_2_over_eta = alpha_2 / self.eta
        n = np.arange(50)  # number of counts

        # Theoretical probability distribution of photon number of dark state
        # Average photon counts from the ion
        p_dark = np.exp(-alpha_1_over_eta * self.lambda_0) * \
                 (alpha_1_over_eta / (1 - alpha_1_over_eta) ** (n + 1) *
                  gammainc(n + 1, (1 - alpha_1_over_eta) * self.lambda_0) + (n == 0))

        # Theoretical probability distribution of photon number of bright state
        # Average photon counts from the ion
        p_bright = np.exp(-(1 + alpha_2_over_eta) * self.lambda_0) * self.lambda_0 ** n / special.factorial(
            n) + alpha_2_over_eta / (
                           1 + alpha_2_over_eta) ** (n + 1) * gammainc(n + 1, (1 + alpha_2_over_eta) * self.lambda_0)

        # Poisson noise distribution
        if self.r_bg is None:
            p_bg = None
        else:
            lambda_bg = np.prod(self.dim) * tau * self.r_bg / 1e-3  # Average photon counts from noise
            p_bg = np.exp(-lambda_bg) * lambda_bg ** n / special.factorial(n)

        return p_bright, p_dark, p_bg

    def _sort_sources(self, *sources):
        """
        Sorts sources into ions and background sources.

        Parameters
        ----------
        sources : list of PhotonSource objects
        """
        self.ions = []
        self.bkg = []
        self.sources = []

        for src in sources:
            if type(src) is Ion:
                self.ions.append(src)
                self.sources.append(src)
            elif type(src) is Bkg:
                self.bkg.append(src)
                self.sources.append(src)
