from IonImaging import delta, gamma, Delta_1, Delta_2
from scipy.special import factorial
from scipy.special._ufuncs import gammainc

from .classes import *
from . import misc, visualize
import numpy as np
import tensorflow.compat.v1 as tf
import tensorflow_probability as tfp
import os
import datetime

tfd = tfp.distributions
tfb = tfp.bijectors

import matplotlib.pyplot as plt


class TFPSimulation(Simulation):
    """
    Simulation using TensorFlow Probability.

    Attributes
    ----------
    dim : list of int
    sources : list of Ion or Bkg objects or None
    tau : float or None
    s : float or None
    na : float or None
    qe : float or None
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
        super().__init__(dim, tau, s, na, qe, r_bg, spot_diam, spacing, angle, ccd_pos, sources)

        self.ion_pos = []
        self.labels = []
        self.samples = []
        self.joint_dist = None

    def gen_images(self,
                   n_images=10,
                   n_testing=0,
                   save_data=False,
                   label='',
                   shuffle=True,
                   verbose=1):
        """
        Generates a set of samples in which the state and position of the ions is randomized for each image.

        For each sample, places any number of ions (as few as half the total number of ions in the experiment) in a
        chain on the CCD. Each sample consists of a calibration image in which all ions emit exactly 100 photons,
        and a state measurement image, where the state is random. For each sample, the positions and states of ions
        are kept.

        Parameters
        ----------
        n_images : int
            Total number of generated images.
        n_testing : int
            Number of images to separate into a testing set. Must be less than n_images.
        save_data : bool
        label : str
            Base name for files to be saved.
        shuffle : bool
            Whether or not to shuffle images before saving.
        verbose : 0 | 1 | 2
            0 : Nothing is printed to the console.
            1 : Prints progress of generating images every 2% complete.
            2 : Prints the state of ions in each image generated.

        Returns
        -------

        """

        states = misc.gen_random_state(len(self.ions), out_type='bin', n_samples=n_images, allow_duplicates=True)

        labels = []
        ion_pos = (-1)*np.ones((n_images, 15, 2))

        for i in range(n_images):
            pos = self.randomize_ion_pos()

            pos = np.array(pos)
            n_ions = pos.shape[0]
            ion_pos[i, :n_ions, :] = pos

            if verbose == 1:
                every = 2
                if (every * n_images // 100) != 0:
                    if (i + 1) % (every * n_images // 100) == 0:
                        print('%0.2f%% Complete' % (100 * (i+1) / n_images))
            elif verbose == 2:
                print('Sampling state %s' % states[i][:n_ions])

            self.sample_state(1, states[i], n_ions)
            labels.append([states[i][:n_ions]])

        imgs_measstate, imgs_calib = self.extract_imgs()

        imgs_measstate = np.array(imgs_measstate).astype('uint8')
        imgs_calib = np.array(imgs_calib).astype('uint8')
        labels = np.array(labels).flatten()

        if shuffle:
            imgs_measstate, imgs_calib, labels, ion_pos = \
                misc.shuffle_together(imgs_measstate, imgs_calib, labels, ion_pos)

        train_measstate = imgs_measstate[n_testing:]
        train_calib = imgs_calib[n_testing:]
        train_labels = labels[n_testing:]
        train_pos = ion_pos[n_testing:]

        test_measstate = imgs_measstate[:n_testing]
        test_calib = imgs_calib[:n_testing]
        test_labels = labels[:n_testing]
        test_pos = ion_pos[:n_testing]

        # test_decay_times = decay_times[:n_testing]
        # test_ion_emission = ion_emission[:n_testing]
        # test_bkg_images = bkg_images[:n_testing]

        if save_data:
            date = datetime.date.today().strftime('%Y%m%d')
            os.makedirs(date, exist_ok=True)

            fname = date + '/' + label + '_' + date

            np.save(fname + '_TrainMeasState.npy', train_measstate, allow_pickle=True)
            np.save(fname + '_TrainCalibration.npy', train_calib, allow_pickle=True)
            np.save(fname + '_TrainLabels.npy', train_labels, allow_pickle=True)
            np.save(fname + '_TrainIonPos.npy', train_pos, allow_pickle=True)

            np.save(fname + '_TestMeasState.npy', test_measstate, allow_pickle=True)
            np.save(fname + '_TestCalibration.npy', test_calib, allow_pickle=True)
            np.save(fname + '_TestLabels.npy', test_labels, allow_pickle=True)
            np.save(fname + '_TestIonPos.npy', test_pos, allow_pickle=True)

            # np.save(fname + '_TestDecayTimes.npy', test_decay_times, allow_pickle=True)
            # np.save(fname + '_TestIonEmission.npy', test_ion_emission, allow_pickle=True)
            # np.save(fname + '_TestBkgImages.npy', test_bkg_images, allow_pickle=True)

    def extract_imgs(self):
        """
        Creates a set of calibration images and a set of corresponding state measurement images, from the list
        of samples previously generated. The order of the images is the same for both sets.

        Returns
        -------
        imgs_measstate : ndarray
            State measurement images
        imgs_calib : ndarray
            Calibration images

        """
        n_samples = len(self.samples)
        imgs_measstate = []
        imgs_calib = []

        for i in range(n_samples):
            imgs_measstate.append(self.samples[i]['img_measstate'].numpy().reshape(self.dim))
            imgs_calib.append(self.samples[i]['img_calib'].numpy().reshape(self.dim))

        return imgs_measstate, imgs_calib

    def sample_state(self, n_trials, state=None, n_ions=None):
        """
        Simulates a given number of trials for an ion imaging experiment. Appends results to self.samples.

        State and positions of ions are the same for all trials in a single call of this method. Creates a dictionary
        containing information about intermediate processes such as decay time, photon emission by ions, as well as
        final images (calibration images and state detection images).

        Parameters
        ----------
        n_trials : int
        state : str
        n_ions : int (optional)
            If not specified, uses all ions in the experiment.

        Returns
        -------
        samples : tfp.Sample object

        """
        if n_ions is None:
            n_ions = len(self.ions)
        elif n_ions > len(self.ions):
            raise ValueError('n_ions cannot exceed number of ions in Simulation')

        if state is None:
            state_array = misc.as_bin_array(self.state_asfloat(), out_type='float')
        else:
            state_array = misc.as_bin_array(state, out_type='float')

        self.labels.append(misc.as_bin_str([state_array[:n_ions]])[0])

        state_array = state_array.astype('float')
        state = state_array[:, :n_ions].flatten().tolist()
        joint_dist = self._create_joint_dist(state, n_ions)
        self.joint_dist = joint_dist
        samples = joint_dist.sample(n_trials, n_ions)
        self.samples.append(samples)

        return samples

    def plot_samples_slider(self):
        """
        Plots the calibration and state detection image for each sample one at a time, providing a slider to select
        which sample to plot.

        Returns
        -------

        """

        fig, (ax1, ax2) = plt.subplots(ncols=2)

        imgs_meas, imgs_cali = self.extract_imgs()

        max_counts_meas = np.max(imgs_meas)
        max_counts_cali = np.max(imgs_cali)

        n_samples = len(self.samples)

        plt.subplots_adjust(bottom=0.20)
        i_src = 0
        ax1.set_title(self.labels[i_src])
        ax2.set_title('Calibration')
        im_meas = ax1.matshow(imgs_meas[i_src], cmap='gray', vmin=0, vmax=max_counts_meas)
        im_cali = ax2.matshow(imgs_cali[i_src], cmap='gray', vmin=0, vmax=max_counts_cali)
        for ax, im in [[ax1, im_meas], [ax2, im_cali]]:
            ax.set_xticks([])
            ax.set_yticks([])
            fig.colorbar(im, ax=ax, label='Photon Counts')

        # cax = plt.axes([0.85, 0.2, 0.05, 0.7])
        # fig.colorbar(im, cax=cax, label='PSF')
        # plt.cm.ScalarMappable(cmap='gray')

        axcolor = 'lightgoldenrodyellow'
        ax_src = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)

        slider = Slider(ax_src, 'Source', 0, n_samples-1, valinit=i_src, valstep=1)

        def update(val):
            i_src = int(slider.val)
            im_meas.set_data(imgs_meas[i_src])
            im_cali.set_data(imgs_cali[i_src])
            ax1.set_title(self.labels[i_src])
            fig.canvas.draw_idle()

        slider.on_changed(update)

        plt.show()


    def _create_joint_dist(self, state, n_ions=None):
        """
        Creates and returns a tensorflow distribution object corresponding to the given state.

        Parameters
        ----------
        state : list of float

        Returns
        -------
        tfd.JointDistributionNamed

        """

        if n_ions is None:
            n_ions = len(self.ions)
        elif n_ions > len(self.ions):
            raise ValueError('n_ions cannot exceed number of ions in Simulation')

        state = state[:n_ions]

        self.update_params()
        decay_rates = self._get_decay_rates(state)
        calibration_counts = np.array([[100.] * n_ions]).astype('float32').flatten()  # reshape((n_ions, 1))
        n_pixels = int(np.prod(self.dim))

        flat_ion_psfs = [self.psf_ions[i].flatten() for i in range(n_ions)]
        flat_bkg_psf = [self.psf_bkg[i].flatten() for i in range(len(self.bkg))]

        joint_dist = tfd.JointDistributionNamed(dict(

            ion_tau_decay=tfd.Exponential(decay_rates),

            ion_t_bright=lambda ion_tau_decay: tfd.Deterministic(
                tf.math.abs((tf.constant(state)) * self.tau - tf.math.minimum(ion_tau_decay, self.tau))),

            ion_emit_measstate=lambda ion_t_bright: tfd.Poisson((ion_t_bright / self.tau) * self.lambda_0),

            ion_emit_calib=tfd.Poisson(tf.constant(calibration_counts)),

            imgs_ion_measstate=lambda ion_emit_measstate: tfd.Multinomial(ion_emit_measstate, probs=flat_ion_psfs),

            imgs_bkg_measstate=tfd.Multinomial(n_pixels * self.r_bg * self.tau / 1e3, probs=flat_bkg_psf),

            imgs_ion_calib=lambda ion_emit_calib: tfd.Multinomial(ion_emit_calib, probs=flat_ion_psfs),

            imgs_bkg_calib=tfd.Multinomial(n_pixels * self.r_bg * self.tau / 1e3, probs=flat_bkg_psf),

            img_measstate=lambda imgs_ion_measstate, imgs_bkg_measstate: tfd.Deterministic(
                tf.math.reduce_sum(imgs_ion_measstate, 1) + tf.math.reduce_sum(imgs_bkg_measstate, 1)),

            img_calib=lambda imgs_ion_calib, imgs_bkg_calib: tfd.Deterministic(
                tf.math.reduce_sum(imgs_ion_calib, 1) + tf.math.reduce_sum(imgs_bkg_calib, 1))
        ))

        return joint_dist

    def _get_decay_rates(self, state):
        """
        Creates a list of decay rates (equal to 1/tau_L), one for each ion.

        Parameters
        ----------
        state : list of float

        Returns
        -------
        list of float
            Decay rate for each ion to be imaged

        """

        tau_L1 = 36 * (Delta_1 ** 2) / (self.s * (gamma ** 3))
        tau_L2 = 36 * (Delta_2 ** 2) / (self.s * (gamma ** 3))

        tau_decay_list = []
        for single_ion_state in state:
            if single_ion_state == 0:
                tau_decay_list.append(1 / tau_L1)
            elif single_ion_state == 1:
                tau_decay_list.append(1 / tau_L2)
            else:
                raise ValueError('Invalid state given')
        return tau_decay_list
