# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 18:50 2020

@author: Scott

Module used for batch classification of state detection images, using either a Linear SVM (one-hot encoding), or a
fully-connected neural network (binary encoding). Currently, the use of calibration images is not implemented here.

"""

import tensorflow as tf
from errors import gen_color_list
from visualize import plot_spots

from Misc.generating_psfs import gen_center_points
from Misc.psf_generator import *
from sklearn import svm
from scipy.special import softmax

keras = tf.keras


def plot_images(imgs, nrows, ncols, labels=None, figsize=(10, 10), spot_kwargs=None, cbar=True):
    """
    Plots given images in a grid of subplots, optionally plotting spots indicating ion positions.


    Parameters
    ----------
    imgs : ndarray
    nrows : int
        Number of rows of images
    ncols : int
        Number of columns of images
    labels : list of str
    figsize : tuple
    spot_kwargs : dict
        Contains xpos, ypos, radius, each a list. See visualize.plot_spots
    cbar : bool

    Returns
    -------

    """
    vmin = 0
    vmax = np.max(imgs)

    imgs = imgs.tolist()
    fig = plt.figure(figsize=figsize)
    axes = []

    if spot_kwargs is not None:
        if 'color_list' in spot_kwargs:
            color_list = spot_kwargs.pop('color_list')
        else:
            color_list = None

    for i in range(nrows * ncols):
        img = np.array(imgs[i])
        ax = plt.subplot(nrows, ncols, i + 1)
        axes.append(ax)

        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        im = plt.imshow(img, cmap=plt.cm.gray, vmin=vmin, vmax=vmax)
        plt.xlabel(labels[i])

        if spot_kwargs is not None:
            if color_list is not None:
                plot_spots(ax, color=color_list[i], **spot_kwargs)
            else:
                plot_spots(ax, **spot_kwargs)

    if cbar:
        fig.subplots_adjust(left=0.2)
        cbar_ax = fig.add_axes([0.92, 0.06, 0.05, 0.7])
        fig.colorbar(im, cax=cbar_ax)

    plt.show()


def binarize_psfs(psf_list):
    """
    Converts point-spread functions into binary functions. Values are mapped to 1 if they are larger than twice the
    mean of all values, mapped to zero otherwise.

    Parameters
    ----------
    psf_list : list of ndarray

    Returns
    -------
    list of ndarray

    """
    psf_shape = psf_list[0].shape
    for i in range(len(psf_list)):
        psf = psf_list[i].flatten()
        binary_psf = np.copy(psf)
        for j in range(psf.size):
            binary_psf[j] = int(psf[j] > 2 * np.average(psf))
        psf_list[i] = binary_psf.reshape(psf_shape)
    return psf_list


def shape_images(images, nx=3):
    """
    Reshapes a batch of images to a given width.

    Parameters
    ----------
    images : array-like
    nx : int
        Number of pixels wide

    Returns
    -------
    ndarray
        Reshaped images

    """

    images = np.array(images)
    n_images = images.shape[0]

    if ((images.size // n_images) % nx) != 0:
        raise ValueError('Incompatible dimension given')

    ny = images.size // n_images // nx
    return np.reshape(images, (n_images, nx, ny))


def flatten_images(images):
    """
    Flattens a batch of images.

    Parameters
    ----------
    images : array-like

    Returns
    -------

    """
    images = np.array(images)
    n_images = images.shape[0]
    n_pixels = images.size // n_images
    return np.reshape(images, (n_images, n_pixels))


def pre_process(data):
    """
    Preprocesses data before passing to classifier. Divides values by 255.

    Parameters
    ----------
    data : ndarray

    Returns
    -------
    ndarray

    """
    data = data / 255.
    return data


def image_from_prediction_array(i, predictions_array, imgs, true_labels=None, vmax=None, class_names=None):
    """
    On the current axis, plots the i-th image in a given set, along with a color-coded label corresponding to whether
    or not the predicted state is correct. Predictions determined by most likely state based on a set of confidence
    levels (percentage of likelihood for each state, using one-hot encoding).

    Parameters
    ----------
    i : int
        Index of image to plot.
    predictions_array : array-like
        1D, n-th value corresponds to the percentage likelhood of the i-th image corresponding to the n-th state.
    imgs : list of ndarray
    true_labels : list of int
        1D, n-th value is the true state of the n-th image
    vmax : float
        Max value represented in colorplot
    class_names : list of str

    Returns
    -------

    """
    img = imgs[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, vmax=vmax, cmap=plt.cm.gray)

    predicted_label = np.argmax(predictions_array)

    xlabel = "{} {:2.0f}%".format(class_names[predicted_label],
                                  100 * np.max(predictions_array))

    if true_labels is not None:
        true_label = true_labels[i]
        xlabel = xlabel + " ({})".format(class_names[true_label])
        if predicted_label == true_label:
            color = 'blue'
        else:
            color = 'red'
    else:
        color = 'black'

    plt.xlabel(xlabel, color=color, fontsize=10)


def image_from_prediction(i, predictions, imgs, true_labels=None, vmax=None, class_names=None):
    """
    On the current axis, plots the i-th image in a given set, along with a color-coded label corresponding to whether
    or not the predicted state is correct.

    Parameters
    ----------
    i : int
        Index of image to plot.
    predictions : array-like
        1D, n-th value is the predicted state of the n-th image
    imgs : list of ndarray
    true_labels : list of int
        1D, n-th value is the true state of the n-th image
    vmax : float
        Max value represented in colorplot
    class_names : list of str

    Returns
    -------

    """
    img = imgs[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, vmax=vmax, cmap=plt.cm.gray)

    predicted_label = predictions[i]

    xlabel = "{}".format(class_names[predicted_label])

    if true_labels is not None:
        true_label = true_labels[i]
        xlabel = xlabel + " ({})".format(class_names[true_label])
        if predicted_label == true_label:
            color = 'blue'
        else:
            color = 'red'
    else:
        color = 'black'

    plt.xlabel(xlabel, color=color, fontsize=10)


def plot_value_array(i, predictions_array, true_labels=None):
    """
    Plots confidence levels that ions are in a particular state (percentage of likelihood) as a bar graph.

    Parameters
    ----------
    i : int
    predictions_array : array-like
        n-th value is the likelihood that ions in image i are in the n-th state
    true_labels : array-like (optional)
        i-th value is the true state of the i-th image

    Returns
    -------

    """
    if true_labels is not None:
        true_label = true_labels[i]
    plt.grid(False)
    n_states = np.size(predictions_array)
    plt.xticks([])
    plt.yticks([])
    bar_graph = plt.bar(range(n_states), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    bar_graph[predicted_label].set_color('red')
    if true_labels is not None:
        bar_graph[true_label].set_color('blue')


def plot_predictions(images, predictions, testing_labels=None, vmax=None, num_rows=5, num_cols=3,
                     class_names=None):
    """
    Plot the first X images in a set, their predicted labels, and the true labels, color coded depending on if the
    prediction is correct.

    Parameters
    ----------
    images : ndarray
    predictions : array-like
        n-th value is the predicted state for the n-th image
    testing_labels : array-like (optional)
        n-th value is the true state for the n-th image
    vmax : float (optional)
    num_rows : int (optional)
        Number of rows of images
    num_cols : int (optional)
        Number of columns of images
    class_names : array-like (optional)

    Returns
    -------

    """

    testing_data = shape_images(images)
    num_images = num_rows * num_cols
    plt.figure(figsize=(2 * num_cols, num_rows))

    if vmax is None:
        vmax = np.max(images[:num_images, :, :])

    if class_names is None:
        misc.decimal_to_binary(*np.arange(int(2 ** np.max(predictions))).tolist())

    for i in range(num_images):
        plt.subplot(num_rows, num_cols, i + 1)
        image_from_prediction(i, predictions, testing_data, testing_labels, vmax, class_names=class_names)
    plt.tight_layout()
    plt.show()


def plot_predictions_arrays(images, predictions, testing_labels=None, vmax=None, num_rows=5, num_cols=3,
                            class_names=None):
    """
    Plot the first X images in a set, their predicted labels, and the true labels, color coded depending on if the
    prediction is correct. A bar graph of the predicted likelihood of each state is plotted next to each image.

    Parameters
    ----------
    images : ndarray
    predictions : ndarray
        Index [i,n] is the likelihood that the ions in the i-th image are in the n-th state
    testing_labels : array-like (optional)
        n-th value is the true state for the n-th image
    vmax : float (optional)
    num_rows : int (optional)
        Number of rows of images
    num_cols : int (optional)
        Number of columns of images
    class_names : array-like (optional)

    Returns
    -------

    """

    testing_data = shape_images(images)
    num_images = num_rows * num_cols
    plt.figure(figsize=(2 * 2 * num_cols, num_rows))

    if vmax is None:
        vmax = np.max(images[:num_images, :, :])

    if class_names is None:
        class_names = np.arange(predictions.shape[1]).astype(str)

    for i in range(num_images):
        plt.subplot(num_rows, 2 * num_cols, 2 * i + 1)
        image_from_prediction_array(i, predictions[i, :], testing_data, testing_labels, vmax, class_names=class_names)
        plt.subplot(num_rows, 2 * num_cols, 2 * i + 2)
        plot_value_array(i, predictions[i, :], testing_labels)
    plt.tight_layout()
    plt.show()


class StateClassifier:
    """
    Class used to train and use either a linear SVM, fully-connected neural network, or an algorithmic classifier.

    Contains of a training set - and optionally a testing set - of images with labels.

    """
    def __init__(self, n_ions, training_images, training_labels, testing_images=None, testing_labels=None,
                 psf_list=None, nn_encoding='binary'):
        """

        Parameters
        ----------
        n_ions : int
        training_images : ndarray
        training_labels : list
        testing_images : ndarray (optional)
        testing_labels : list
        psf_list : list
            PSF of all ions
        nn_encoding : str
            'binary' or 'onehot'
        """
        self.n_ions = n_ions
        self.image_shape = training_images.shape[1:]
        self.ny_pixels, self.nx_pixels = training_images.shape[1:]
        self.nx_pixels = self.nx_pixels // self.n_ions

        self.training_images = training_images
        if type(training_labels) is np.ndarray:
            self.training_labels = training_labels.tolist()
        else:
            self.training_labels = training_labels

        self.testing_images = testing_images
        if type(testing_labels) is np.ndarray:
            self.testing_labels = testing_labels.tolist()
        else:
            self.testing_labels = testing_labels

        self.model = None
        self.nn_encoding = nn_encoding
        self.clf = None
        self.thresholds = None

        if psf_list is not None:
            self.psf_list = psf_list

    def train_neural_net(self, epochs=10, encoding=None):
        """
        Creates a keras model for a fully-connected feed forward neural network, and trains the model.

        Parameters
        ----------
        epochs : int
        encoding : str (optional)
            'onehot' or 'binary'

        Returns
        -------
        keras.Model

        """
        training_data = pre_process(self.training_images)
        if encoding is not None:
            self.nn_encoding = encoding

        input_layers = [keras.layers.Flatten(input_shape=(self.ny_pixels, self.nx_pixels * self.n_ions))]

        if self.nn_encoding == 'onehot':
            training_labels = misc.as_dec_array(self.training_labels)

            hidden_layers = [keras.layers.Dense(64, activation='selu')]
            output_layers = [keras.layers.Dense(int(2 ** self.n_ions))]

            loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),

        elif self.nn_encoding == 'binary':
            training_labels = misc.as_bin_array(self.training_labels)

            hidden_layers = [keras.layers.Dense(200, activation='softsign')]
            output_layers = [keras.layers.Dense(self.n_ions)]

            loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)

        else:
            raise ValueError('Invalid encoding specified')

        layers = input_layers + hidden_layers + output_layers

        model = keras.Sequential(layers)

        model.compile(optimizer='adam',
                      loss=loss,
                      metrics=['accuracy'])

        model.fit(training_data, training_labels, epochs=epochs, verbose=0)

        self.model = model

        return model

    def neural_net_predict(self, testing_images=None, testing_labels=None, plot=True):
        """
        Uses a fully-connected feed forward neural network to predict the state of each image in a set.

        The keras model must have been previously created using the train_neural_net method.

        Parameters
        ----------
        testing_images : ndarray
        testing_labels : list
        plot : bool

        Returns
        -------
        ndarray

        """
        if testing_images is None:
            if self.testing_images is not None:
                testing_data = pre_process(self.testing_images)
                testing_labels = self.testing_labels

        if self.nn_encoding == 'onehot':
            probability_model = tf.keras.Sequential([self.model,
                                                     tf.keras.layers.Softmax()])
            predictions = probability_model.predict(testing_data)
            most_probable_states = np.argmax(predictions, axis=1).tolist()

        elif self.nn_encoding == 'binary':
            probability_model = tf.keras.Sequential([self.model,
                                                     tf.keras.layers.Activation('sigmoid')])
            predictions = probability_model.predict(testing_data)
            most_probable_states = np.round(predictions).tolist()

        if testing_labels is not None:
            predictions_bin_str = np.array(misc.as_bin_str(most_probable_states))
            true_states_bin_str = np.array(misc.as_bin_str(testing_labels))
            n_correct = (predictions_bin_str == true_states_bin_str).astype(int).sum()
            accuracy = float(n_correct) / len(testing_labels)
            print('Neural Net: Accuracy = %0.2f%%' % (100 * accuracy))

        if plot:
            plot_predictions_arrays(self.testing_images, predictions, self.testing_labels, class_names=self.class_names)

        return predictions

    def train_svm(self, max_iter=1000):
        """
        Creates classifier using a linear support vector classification.

        Parameters
        ----------
        max_iter : int

        Returns
        -------
        svm.LinearSVC

        """
        training_data = flatten_images(self.training_images)
        clf = svm.LinearSVC(max_iter=max_iter)
        clf.fit(training_data, self.training_labels)
        self.clf = clf

        return clf

    def svm_predict(self, testing_images=None, testing_labels=None, plot=True):
        """
        Uses a linear SVC to predict the state of each image in a set.

        The SVC must have been previously created using the train_svm method.

        Parameters
        ----------
        testing_images : ndarray
        testing_labels : list
        plot : bool

        Returns
        -------
        ndarray
            State predictions

        """
        if testing_images is None:
            testing_data = self.testing_images
            testing_labels = self.testing_labels

        testing_data = flatten_images(testing_data)

        predictions = softmax(self.clf.decision_function(testing_data), axis=1)

        if testing_labels is not None:
            accuracy = self.clf.score(testing_data, testing_labels)
            print('Linear SVM: Accuracy = %0.2f%%' % (100 * accuracy))

        if plot:
            plot_predictions_arrays(self.testing_images, predictions, self.testing_labels, class_names=self.class_names)

        return predictions

    def optimize_thresholds(self):
        """
        For each ion, determines the threshold with which to compare the sum of photon counts (weighted by each ion's
        PSF) to maximize state measurement fidelity.

        Returns
        -------
        list
            Optimal threshold for each ion.

        """
        if self.psf_list is None:
            self.psf_list = self.approx_psfs(subtract_bkg=True)
        psf_list = binarize_psfs(self.psf_list)

        ideal_thresh_list = []
        for i in range(self.n_ions):
            print('Running ion %i of %i' % (i + 1, self.n_ions))
            psf = psf_list[i]
            actual_states = misc.decimal_to_binary(*self.training_labels.tolist(), n_bits=self.n_ions, get_digit=i)
            actual_states = np.array(actual_states).astype(int)
            high_score = 0
            ideal_thresh_list.append(0)
            for thresh in range(15):
                predicted_states = self.single_ion_thresh_predict(self.training_images, thresh, psf)
                predicted_states = np.array(predicted_states)
                n_correct = np.sum((predicted_states == actual_states).astype(int))

                if n_correct > high_score:
                    high_score = n_correct
                    ideal_thresh_list[-1] = thresh
        self.thresholds = ideal_thresh_list
        return ideal_thresh_list

    def threshold_predict(self, testing_images=None, testing_labels=None, plot=True):
        """
        Predicts the state of all ions in an image by weighting the photon counts in each pixel by the value of the
        PSF of each ion, and comparing the weighted sum to a threshold for each ion. Thresholds for each ion must
        have already been set using the optimize_thresholds method or by setting them manually.

        Parameters
        ----------
        testing_images : ndarray
        testing_labels : list
        plot : bool

        Returns
        -------
        ndarray
            State predictions

        """
        if testing_images is None:
            testing_data = self.testing_images
            testing_labels = self.testing_labels

        predictions = np.zeros(np.shape(testing_data)[0], dtype=int)
        for i in range(self.n_ions):
            predicted_single_states = self.single_ion_thresh_predict(testing_data, self.thresholds[i], self.psf_list[i])
            predicted_single_states = np.array(predicted_single_states)
            predictions = predictions + predicted_single_states * int(2 ** i)

        if testing_labels is not None:
            correct_guesses = (predictions == testing_labels).astype(int)
            accuracy = np.average(correct_guesses)
            print('Threshold Method: Accuracy = %0.2f%%' % (100 * accuracy))

        if plot:
            plot_predictions(testing_data, predictions, testing_labels=testing_labels, class_names=self.class_names)

        return predictions

    def single_ion_thresh_predict(self, imgs, threshold, psf):
        """
        Predicts the state of a single ion, weighting counts for each ion according to a given PSF, and comparing the
        weighted sum of photon counts to a given threshold.

        Parameters
        ----------
        imgs : array-like
        threshold : int
        psf : array-like
            Shape should match that of each image.

        Returns
        -------

        """
        imgs = np.array(imgs)
        predictions = []
        for i in range(imgs.shape[0]):
            weighted_counts = np.sum(psf * imgs[i], axis=None)
            predictions.append(weighted_counts > threshold)
        return predictions

    def approx_psfs(self, subtract_bkg=True):
        """
        Approximates the PSF of each ion.

        Training images must consist of at least one image per ion where all ions are dark except for one.
        e.g. for a 3 ion chain, training images must consist of at least one image each where the ions are in state
        100, 010, and 001

        Parameters
        ----------
        subtract_bkg : bool
            If True, sets the minimum PSF value for each ion to zero.

        Returns
        -------
        list of ndarray
            Approximated PSF of each ion.

        """
        psf_list = []

        labels_as_str = np.array(misc.as_bin_str(self.training_labels))

        for i in range(self.n_ions):
            state = '0' * i + '1' + '0' * (self.n_ions - i - 1)
            images = self.training_images[state == labels_as_str]
            psf_list.append(np.average(images, axis=0))

        if subtract_bkg:
            for i in range(len(psf_list)):
                psf_list[i] = psf_list[i] - psf_list[i].min()

        for i in range(len(psf_list)):
            psf_list[i] = psf_list[i] / psf_list[i].sum()

        return psf_list

    def plot_samples(self, n_samples, nrows=None, ncols=3, plot_spots=False):
        """
        Plots the first X training samples.

        Parameters
        ----------
        n_samples : int
        nrows : int
            Number of rows of subplots
        ncols : int
            Number of columns of subplots
        plot_spots : bool
            Whether or not to plot circles representing ion positions

        Returns
        -------

        """
        if nrows is None:
            nrows = (n_samples + (n_samples % ncols)) // 3
        imgs = self.training_images[:n_samples]
        labels = np.array(self.class_names)[self.training_labels[:n_samples].astype(int)]
        # labels = self.training_labels[:n_samples]

        extent = [0, self.n_ions * self.nx_pixels, 0, self.ny_pixels]
        centers = gen_center_points(self.n_ions, 0.95 * 2, extent[1] / 2.)

        color_list = gen_color_list(self.training_labels[:n_samples], self.n_ions)

        if plot_spots:
            spot_kwargs = {
                'centers': centers - 0.5,
                'radius': 1.0,
                'y_pos': self.ny_pixels / 2 - 0.5,
                'color_list': color_list
            }
        else:
            spot_kwargs = None

        plot_images(imgs, nrows, ncols, labels=labels, spot_kwargs=spot_kwargs)
