# -*- coding: utf-8 -*-
"""
Created on Sat Aug 15 21:19 2020

@author: Scott

Used to compare predictions and true ion states, analyzing the cause of errors. Currently optimize/only works for 5 ion
images.

"""

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy
import numpy as np
from matplotlib import pyplot

from . import misc, visualize
from .classification import binarize_psfs


def autolabel(rects, ax=None):
    """Attach a text label above each bar in rects, displaying its height.

    Parameters
    ----------
    rects : list
        matplotlib.patches.Rectangle objects
    ax : matplotlib.Axes (optional)

    """

    if ax is None:
        ax = plt.gca()

    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)


def get_indv_ion_states(states, n_ions):
    """
    Converts decimal integers representing the state of an ion chain to binary.

    Parameters
    ----------
    states : list
        Decimal numbers representing states of ion chain
    n_ions : int
        Number of ions in the experiment

    Returns
    -------
    array

    """
    n_states = int(2 ** n_ions)
    n_trials = len(states)

    indv_ion_states = np.zeros((n_trials, n_ions))
    for i in range(n_trials):
        tally = states[i]
        for j in range(n_ions):
            indv_ion_states[i, n_ions - j - 1] = tally % 2
            tally = tally // 2

    indv_ion_states = indv_ion_states.flatten()

    return indv_ion_states


class ErrorAnalysis:
    """
    Used to study statistics of incorrect guesses.

    """

    def __init__(self, n_ions, predictions, true_states, psf_list=None, decay_times=None):
        """
        Parameters
        ----------
        n_ions : int
        predictions : array-like
        true_states : array-like
        decay_times : ndarray
            i,j: Decay time of j-th ion in i-th trial
        """

        self.n_ions = n_ions
        self.predictions = misc.as_bin_array(predictions, out_type='int', n_bits=n_ions)
        self.true_states = misc.as_bin_array(true_states, out_type='int', n_bits=n_ions)
        self.psf_list = psf_list
        self.decay_times = decay_times
        self.n_samples = self.predictions.shape[0]

        full_state_acc = self.get_full_state_accuracy()
        print('Full State Accuracy = %0.2f%%' % (100 * full_state_acc))

    def get_indv_ion_accuracies(self):
        """
        Determines the average accuracy of predictions for each ion, separately.

        Returns
        -------
        array
            i: Average accuracy of i-th ion

        """
        return (self.predictions == self.true_states).astype('float').sum(axis=0) / self.n_samples

    def plot_indv_ion_accuracies(self, **kwargs):
        """
        Plots the average accuracy of each ion in a bar graph.

        Parameters
        ----------
        kwargs : dict
            Passed into ax.bar

        Returns
        -------
        array
            i: Average accuracy of i-th ion

        """
        accuracies = self.get_indv_ion_accuracies()

        fig, ax = plt.subplots()
        ax.bar(np.arange(self.n_ions) + 1, accuracies, **kwargs)
        ax.set_ylim([0.95, 1.0])
        plt.ylabel('Accuracy')
        plt.xlabel('Ion')
        plt.show()

        return accuracies

    def plot_indv_state_accuracies(self, **kwargs):
        """
        Plots the average accuracy of ions prepared dark vs those prepared bright as a bar graph.

        Parameters
        ----------
        kwargs : dict
            Passed into plt.bar

        Returns
        -------
        acc_bright : float
        acc_dark : float

        """
        acc_bright, acc_dark = self.get_indv_state_accuracies()

        fig, ax = plt.subplots()
        ax.bar([1, 2], [acc_dark, acc_bright], **kwargs)
        plt.ylabel('Accuracy')
        plt.ylim([0.95, 1.0])
        plt.xticks([1, 2], ['Prepared Dark', 'Prepared Bright'])
        plt.show()

        return acc_bright, acc_dark

    def plot_crosstalks(self, **kwargs):
        """
        Plots the crosstalk of each ion, currently measured as the number of non-zero elements in the PSF of each.

        Parameters
        ----------
        kwargs : dict
            Passed into plt.bar

        Returns
        -------
        list
            Measure of crosstalk for each ion

        """
        crosstalk = self.quantify_crosstalk_somehow()

        fig, ax = plt.subplots()
        ax.bar(np.arange(self.n_ions) + 1, crosstalk, **kwargs)
        # ax.set_ylim([0.95,1.01])
        plt.show()

        return crosstalk

    def get_indv_state_accuracies(self):
        """
        Determines the average accuracy of ions prepared dark and those prepared bright.

        Returns
        -------
        acc_bright : float
        acc_dark : float

        """
        matching = (self.predictions == self.true_states).astype(int)
        acc_bright = np.average(matching[self.true_states == 1])
        acc_dark = np.average(matching[self.true_states == 0])
        return acc_bright, acc_dark

    def quantify_crosstalk_somehow(self):
        """
        Quantifies crosstalk of each ion.

        Currently determines the number of non-zero elements in the PSF of each ion. This is not a true measure of
        crosstalk, but ions whose PSF is more spread out have more crosstalk with neighbouring ions.

        Returns
        -------
        list
            Number of non-zero elements in the PSF of each ion

        """
        n_non_zero = [np.size(psf[psf > 0]) for psf in self.psf_list]
        return n_non_zero

    def get_full_state_accuracy(self):
        """
        Determines the accuracy of full state predictions, i.e. the frequency of correctly predicting the state of all
        ions in the chain.

        Returns
        -------
        acc : float

        """
        predictions = np.array(misc.as_bin_str(self.predictions))
        true_states = np.array(misc.as_bin_str(self.true_states))

        acc = (predictions == true_states).astype('float').sum() / predictions.size

        return acc


def compare_binary_digits(num1, num2, n_digits, check='match'):
    """
    Converts two decimal numbers to binary and compares digits which do not match.

    Parameters
    ----------
    num1 : int, str, or array-like
        Single value, either decimal or binary
    num2 : int, str, or array-like
        Single value, either decimal or binary
    n_digits : int
    check : 'match' | 'mismatch'

    Returns
    -------
    list
        Indices of digits which match/mismatch

    """
    bin_num1, = misc.as_bin_str([num1])
    bin_num2, = misc.as_bin_str([num2])

    digit_indices = []
    for i in range(n_digits):
        if bin_num1[i] == bin_num2[i]:
            if check == 'match':
                digit_indices.append(i)
        else:
            if check == 'mismatch':
                digit_indices.append(i)

    return digit_indices


def get_good_ions(predictions, true_states, n_ions):
    """
    Finds the indices of individual ions for which the state was correctly predicted.

    Parameters
    ----------
    predictions : list
    true_states : list
    n_ions : int

    Returns
    -------
    img_indices : list
        Essentially just np.arange(len(predictions))
    ion_indices : list
        0 : Indices of ions correctly predicted in 1st image
        1 : Indices of ions correctly predicted in 2nd image
        etc.

    """

    img_indices = []
    ion_indices = []
    for i in range(len(predictions)):
        img_indices.append(i)
        ion_indices.append(compare_binary_digits(predictions[i], true_states[i], n_digits=n_ions, check='match'))

    return img_indices, ion_indices


def get_problem_ions(predictions, true_states, n_ions):
    """
    Finds the indices of individual ions for which the state was incorrectly predicted.

    Parameters
    ----------
    predictions : list
    true_states : list
    n_ions : int

    Returns
    -------
    img_indices : list
        Essentially just np.arange(len(predictions))
    ion_indices : list
        0 : Indices of ions incorrectly predicted in 1st image
        1 : Indices of ions incorrectly predicted in 2nd image
        etc.

    """

    img_indices = []
    ion_indices = []
    for i in range(len(predictions)):
        if predictions[i] != true_states[i]:
            img_indices.append(i)
            ion_indices.append(compare_binary_digits(predictions[i], true_states[i], n_ions, check='mismatch'))

    return img_indices, ion_indices


def extract_data(data, img_indices, ion_indices, cap_max=None):
    """
    Given a value for each ion for each trial, returns a list of values according to given indices.

    Parameters
    ----------
    data : ndarray
        i,j : Value for j-th ion in i-th image.
    img_indices : list of int
    ion_indices : list of int
    cap_max : float (optional)
        If values are found bigger than cap_max, sets those values equal to cap_max

    Returns
    -------

    """
    extracted_vals = []
    for i in range(len(img_indices)):
        ii = img_indices[i]
        for j in ion_indices[i]:
            extracted_vals.append(data[ii, j])

    if cap_max is not None:
        for i in range(len(extracted_vals)):
            if extracted_vals[i] > cap_max:
                extracted_vals[i] = cap_max

    return np.array(extracted_vals)


def comp_accuracies_decaytimes(predictions, true_states, decay_times, plot=True):
    """
    Plots a bar graph of the accuracy of state prediction on individual ions as a function of the decay time of the
    ions.

    Sorts decay times into 20 microsecond bins and finds the average accuracy of each ion which decayed within that time
    interval.

    Parameters
    ----------
    predictions : list
        i-th value is the state prediction of the i-th trial
    true_states : list
        i-th value is the state prediction of the i-th trial
    decay_times : ndarray
        i,j : Decay time of j-th ion in i-th trial

    Returns
    -------

    """
    n_ions = np.shape(decay_times)[1]

    decay_times_good = 1e6 * extract_data(decay_times, *get_good_ions(predictions, true_states, n_ions), cap_max=500e-6)
    decay_times_bad = 1e6 * extract_data(decay_times, *get_problem_ions(predictions, true_states, n_ions),
                                         cap_max=500e-6)

    bins = np.linspace(0, 500 - 1e-6, 20)

    num_correct, x_vals = np.histogram(decay_times_good, bins)
    num_incorrect, _ = np.histogram(decay_times_bad, bins)
    x_vals = x_vals[:-1]
    width = (x_vals[1] - x_vals[0])

    accuracies = num_correct / (num_correct + num_incorrect)

    if plot:
        plt.figure()
        plt.bar(x_vals, accuracies, align='edge', width=width)
        plt.figure()
        incorrect_bars = plt.bar(x_vals, num_incorrect, align='edge', width=width)
        autolabel(incorrect_bars)
        plt.show()


def comp_classifiers(predictions, true_labels):
    """
    Compares the full-state accuracy of three methods of predicting ion states: fully-connected feed forward network,
    linear SVM, and threshold using weighted pixel counts by PSF.

    Parameters
    ----------
    predictions : list of array-like
        Neural network predictions, linear SVM predictions, threshold predictions.
    true_labels : array-like

    Returns
    -------

    """
    predictions = np.array(predictions)
    n_classifiers, n_trials = predictions.shape
    score_array = np.zeros((n_trials, n_classifiers))

    for i in range(n_trials):
        for j in range(n_classifiers):
            score_array[i, j] = int(predictions[j, i] == true_labels[i])

    score_as_binary = np.zeros(n_trials, dtype=int)
    for j in range(n_classifiers):
        score_as_binary = score_as_binary + (2 ** j) * score_array[:, n_classifiers - j - 1]

    n_possibilities = int(2 ** n_classifiers)
    labels = ['III', 'IIC', 'ICI', 'ICC', 'CII', 'CIC', 'CCI', 'CCC']

    print('Relative Frequencies:')
    for i in range(n_possibilities):
        n_matching = np.sum((score_as_binary == i).astype(int))
        print('%s: %0.2f%%' % (labels[i], 100 * n_matching / n_trials))


def comp_classifiers_indv_ions(predictions, true_labels, n_ions):
    """
    Compares the accuracy of three methods of predicting states of individual ions: fully-connected feed forward
    network, linear SVM, and threshold using weighted pixel counts by PSF.

    Parameters
    ----------
    predictions : list of array-like
        Neural network predictions, linear SVM predictions, threshold predictions.
    true_labels : array-like

    Returns
    -------

    """
    predictions = np.array(predictions)
    n_classifiers, n_trials = predictions.shape

    n_comps = n_trials * n_ions
    indv_ion_predictions = np.zeros((n_classifiers, n_comps))

    indv_ion_true_states = get_indv_ion_states(true_labels, n_ions)
    for j in range(n_classifiers):
        indv_ion_predictions[j, :] = get_indv_ion_states(predictions[j, :], n_ions)

    comp_classifiers(indv_ion_predictions, indv_ion_true_states)


def gen_color_list(states, n_ions):
    """
    Creates a list of colors, either red or yellow, in which to plot circles to show ion positions. Colours correspond
    to whether or not the state of each ion was correctly predicted.

    Parameters
    ----------
    states : int or array
        State of ions, as decimal or binary
    n_ions : int

    Returns
    -------
    list of str

    """
    color_list = []
    states_binary = misc.decimal_to_binary(*states)
    for i in range(len(states)):
        color_list.append('')
        for j in range(n_ions):
            digit = states_binary[i][j]
            if digit == '1':
                new_color = 'r'
            elif digit == '0':
                new_color = 'y'
            color_list[-1] = color_list[-1] + new_color
    return color_list


def plot_confusion_matrix(predictions, true_labels, class_names, title=None):
    """
    Calculates and plots the confusion matrix for a data set.

    Used to study the most frequent predicted state for each true state.

    Parameters
    ----------
    predictions : list
    true_labels : list
    class_names : list
    title : str

    Returns
    -------
    ndarray
        Confusion matrix

    """
    n_classes = len(class_names)
    n_trials = predictions.size
    conf_mat = np.zeros((n_classes, n_classes))
    for i in range(n_trials):
        conf_mat[predictions[i], true_labels[i]] += 1
    for i in range(n_classes):
        if conf_mat[:, i].sum() != 0:
            conf_mat[:, i] = conf_mat[:, i] / conf_mat[:, i].sum()

    plt.figure()
    im = plt.matshow(conf_mat)
    cbar = plt.colorbar(im, label='Frequency')
    cbar.ax.tick_params(labelsize=10)
    plt.xlabel('True State')
    plt.ylabel('Predicted State')
    if title is not None:
        plt.title(title)
    plt.show()

    return conf_mat
