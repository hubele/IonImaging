# -*- coding: utf-8 -*-
"""


Created on Tue Aug 04 17:55 2020

@author: Scott Hubele
"""

import numpy as np
from IonImaging import misc
from matplotlib import pyplot as plt, gridspec as gridspec

def psf_from_pos(pos, ccd_shape, spot_diam, extent=None, padded_width=None):
    """
    From a set of positions, generates point spread functions for each ion.

    Parameters
    ----------
    pos : list of array-like
        Coordinates of ions (x, y), same dimension as extent
    extent : list
        x_min, x_max, y_min, y_max
        Position of CCD in space
    ccd_shape : list of int
        width, height (pixels)
    spot_diam : float
    padded_width : int
        If specified, pads output list with None values to get length equal to padded_width

    Returns
    -------
    list

    """

    pos = np.array(pos)
    x_pos = pos[:, 0]
    y_pos = pos[:, 1]

    nx_pixels, ny_pixels = ccd_shape

    x_pos = x_pos - x_pos.mean() + nx_pixels / 2.
    y_pos = y_pos - y_pos.mean() + ny_pixels / 2.

    psf = []

    if extent is None:
        extent = [0, nx_pixels, 0, ny_pixels]

    delta_x = (extent[1] - extent[0]) / float(nx_pixels)
    delta_y = (extent[3] - extent[2]) / float(ny_pixels)

    for spot in range(pos.shape[0]):
        psf.append(np.zeros((ny_pixels, nx_pixels)))
        for i in range(ny_pixels):
            for j in range(nx_pixels):
                xmin = float(extent[0]) + j * delta_x
                xmax = xmin + delta_x

                ymin = float(extent[2]) + i * delta_y
                ymax = ymin + delta_y

                pts = misc.gen_rect_mesh([10, 10], [xmin, xmax, ymin, ymax])

                n_in_spot = 0
                for point in pts:
                    if misc.is_in_circle(point, [x_pos[spot], y_pos[spot]], spot_diam / 2.):
                        n_in_spot += 1

                psf[-1][i, j] = n_in_spot / 100.

        psf[-1] = psf[-1] / np.sum(psf[-1])

    if padded_width is not None:
        while padded_width > len(psf):
            psf.append(None)

    return psf


def psf_ion_chain(n_ions, spot_diam=2., spacing=2., angle=0., ccd_dim=None, center=None, plot=False, plot_grid=False,
                  plot_spots=False):
    """
    Generates a set of positions on the CCD

    Parameters
    ----------

    n_ions : int
    spot_diam : float (optional)
        Spot diameter, in pixels.
    spacing : float (optional)
        Centre to centre spacing of spots from adjacent ions.
    angle : float (optional)
        In degrees
    ccd_dim : list (optional)
    center : list (optional)
    plot : bool (optional)
    plot_grid : bool (optional)
    plot_spots : bool (optional)

    Returns
    -------
    list

    """
    delta_x = np.cos(np.pi * angle / 180.) * spacing
    delta_y = np.sin(np.pi * angle / 180.) * spacing

    if ccd_dim is None:
        min_width = delta_x * (n_ions - 1) + spot_diam
        min_height = delta_y * (n_ions - 1) + spot_diam

        nx_pixels = int(np.ceil(min_width))
        ny_pixels = int(np.ceil(min_height))

        ccd_dim = [nx_pixels, ny_pixels]
    else:
        ny_pixels, nx_pixels = ccd_dim

    if center is None:
        center = [nx_pixels / 2., ny_pixels / 2.]

    x_pos = np.arange(n_ions) * delta_x
    y_pos = np.arange(n_ions) * delta_y

    x_pos = x_pos - x_pos.mean() + center[0]
    y_pos = y_pos - y_pos.mean() + center[1]

    pos = np.array([x_pos, y_pos]).transpose()

    extent = [0, nx_pixels, 0, ny_pixels]

    psf = psf_from_pos(pos, extent, ccd_dim, spot_diam)

    if plot or plot_grid or plot_spots:
        fig = plt.figure(figsize=(5, 1.3 * n_ions))
        gs_main = gridspec.GridSpec(1, 2, fig, width_ratios=[20, 1])
        gs_sub = gridspec.GridSpecFromSubplotSpec(n_ions, 1, subplot_spec=gs_main[0])

        for ion in range(n_ions):
            ax = fig.add_subplot(gs_sub[ion, 0])
            im = ax.matshow(psf[ion], extent=extent, vmin=0, vmax=np.max(psf), origin='lower')  # , aspect='auto')
            plt.xticks([])
            plt.yticks([])

            if plot_spots:
                ax.add_artist(plt.Circle((x_pos[ion], y_pos[ion]), spot_diam / 2., fill=False, color='w'))

            if plot_grid:
                for _x in range(1, extent[1]):
                    plt.axvline(_x, color='w', ls='--')
                for _y in range(1, extent[3]):
                    plt.axhline(_y, color='w', ls='--')

        cbar_ax = fig.add_subplot(gs_main[0, 1])
        plt.colorbar(im, cax=cbar_ax, label='Point Spread Function')
        plt.show()

    return psf


