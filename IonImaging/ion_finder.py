"""
Used to determine the positions of ions using "calibration images", using the Doppler cooling beam.

Created on Mon Sep 14 11:50 2020

@author: Scott Hubele

"""

import numpy as np


def gen_dist_matrix(pos1, pos2):
    """
    For two sets of positions, generates a matrix where element i,j is the distance between the i-th position in the
    first set and the j-th position of the second set.

    Parameters
    ----------
    pos1 : array-like
    pos2 : array-like

    Returns
    -------

    """
    pos1, pos2 = np.array(pos1), np.array(pos2)

    dist_mat = np.zeros((pos1.shape[0], pos2.shape[0]))

    for i in range(pos1.shape[0]):
        for j in range(pos2.shape[0]):
            dist_mat[i, j] = np.sqrt(np.sum((pos1[i, :] - pos2[j, :]) ** 2))

    return dist_mat


def convolve_2d(image, kernel):
    """
    Convolves image with a 3x3 kernel. Pads image with zeros so that output matches shape of input.

    Parameters
    ----------
    image : array-like
    kernel : array-like

    Returns
    -------
    ndarray

    """
    image = np.array(image)
    kernel = np.array(kernel)

    img_shape = np.array(image.shape)
    kernel_shape = np.array(kernel.shape)

    img_padded = np.zeros(img_shape + 2)
    img_padded[1:-1, 1:-1] = image

    img_out = np.zeros(img_shape)

    for i in range(img_shape[0]):
        for j in range(img_shape[1]):
            window = img_padded[i:(i + kernel_shape[0]), j:(j + kernel_shape[1])]
            img_out[i, j] = np.sum(kernel * window)

    return img_out


def peak_finder(image, threshold=5, min_separation=1.5):
    """
    Estimates the position of peaks within an image.

    Analyzes regions of 3x3 pixels within the image to find regions where the intensity of the central pixel is
    higher than surrounding pixels by a certain threshold. If a peak is found using this process, the position of the
    peak is estimated from the two pixels in the 3x3 square with the highest intensity. If two peaks are found with
    separation smaller than the specified minimum value, then instead a single peak is recorded in the middle of the
    two identified peaks.

    Parameters
    ----------
    image : array-like
    threshold : int
    min_separation : float

    Returns
    -------
    list
        Estimated peak positions

    """
    image = np.array(image)
    img_shape = np.array(image.shape)

    img_padded = np.zeros(img_shape + 2)
    img_padded[1:-1, 1:-1] = image

    peak_pos_list = []

    for i in range(img_shape[0]):
        for j in range(img_shape[1]):
            window = img_padded[i:(i + 3), j:(j + 3)]

            center_pixel = window[1, 1]

            vals_argsort = np.argsort(window.flatten())

            if vals_argsort[-2] != 4: # Ensure 2nd highest value is not the central pixel in the 3x3 window
                # Surrounding pixels EXCEPT 2nd highest value:
                surrounding_pixels = np.delete(window.flatten(), [4, vals_argsort[-2]])
                second_highest = window.flatten()[vals_argsort[-2]]

                if np.all(center_pixel > (surrounding_pixels + threshold)):
                    ratio = center_pixel / (center_pixel + second_highest)
                    x_pos = ratio * j + (1. - ratio) * (j + vals_argsort[-2] % 3 - 1)
                    y_pos = ratio * i + (1. - ratio) * (i + vals_argsort[-2] // 3 - 1)
                    peak_pos_list.append([x_pos, y_pos])

    dist_mat = gen_dist_matrix(peak_pos_list, peak_pos_list)

    bad_pairs = np.argwhere((dist_mat != 0) & (dist_mat < min_separation)).tolist()

    new_pos_list = []

    for indices in bad_pairs:
        if indices[0] > indices[1]:
            pos1 = peak_pos_list[indices[0]]
            pos2 = peak_pos_list[indices[1]]

            new_pos_list.append([(pos1[0] + pos2[0]) / 2., (pos1[1] + pos2[1]) / 2.])

    bad_detections = np.unique(np.array(bad_pairs).flatten())

    for i in np.sort(bad_detections)[::-1]:
        del (peak_pos_list[i])

    peak_pos_list = peak_pos_list + new_pos_list

    return peak_pos_list


def clean_image(image):
    """
    Removes background noise from an image by clearing pixels outside the region of interest, which is characterized by
    a region with a large number of high intensity pixels.

    Parameters
    ----------
    image : array-like

    Returns
    -------
    ndarray
        Cleaned image.

    """
    image = np.array(image)
    image = np.where(image >= 0, image, 0)

    avg_val = np.average(image)
    img_shape = np.array(image.shape)

    img_padded = np.zeros(img_shape + 2)
    img_padded[1:-1, 1:-1] = image

    img_clean = np.zeros(img_shape)

    for i in range(img_shape[0]):
        for j in range(img_shape[1]):
            window = img_padded[i:(i + 3), j:(j + 3)]

            if np.sum(window) > avg_val * 9 * 2:
                img_clean[i, j] = image[i, j]
            else:
                img_clean[i, j] = 0

    return img_clean


def rem_invalid_pos(sample_set):
    """
    Removes positions in a given list which have values less than 0.

    Positions less than 0 are used to indicate that ions were not imaged in a particular sample, i.e. not included in
    the experiment, and shouldn't be considered.

    Parameters
    ----------
    sample_set : ndarray

    Returns
    -------
    list

    """
    n_samples = sample_set.shape[0]
    n_ions = sample_set.shape[1]

    processed_samples = []
    for i in range(n_samples):
        processed_samples.append([])
        for j in range(n_ions):
            if np.all(sample_set[i, j, :] >= 0):
                processed_samples[-1].append(sample_set[i, j, :].tolist())

    return processed_samples


def gen_rnn_inputs(image, pos_list, n=3):
    """
    Zooms in on a region of n x n pixel values, centered at each given position.

    Parameters
    ----------
    image : array-like
    pos_list : array-like
    n : int

    Returns
    -------

    """

    image = np.array(image)
    pos_list = np.array(pos_list)

    n_pos = pos_list.shape[0]

    rnn_inputs = []

    for i in range(n_pos):
        center = pos_list[i, :]

        x_min, y_min = np.round(center - 1.0).astype(int)

        rnn_inputs.append(image[y_min:(y_min + n), x_min:(x_min + n)])

    return rnn_inputs