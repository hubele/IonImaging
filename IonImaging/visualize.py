import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider
from mpl_toolkits.mplot3d import Axes3D


def plot_imgs(data, exp, shape, show_vals=False, cmap='viridis', hide_axes=True, plot_cbar=True, sigdigs=2, cbar_label=None,
              vmin=None, vmax=None, title=None):
    """
    Plots a set of images as subplots.

    Parameters
    ----------
    cbar_label : str
    data : list of 2d arrays
    exp : Simulation object
    shape : tuple
        nrows, ncols
        Shape of subplots
    show_vals : bool
    sigdigs : int
    hide_axes : bool
    plot_cbar : bool
    cmap : str
    """

    # Shouldn't need to pass in entire Experiment object, just Experiment.sources
    data = np.array(data).tolist()

    # Plot given lists of data

    _vmin, _vmax = get_extrema(data)
    if vmin is None:
        vmin = _vmin
    if vmax is None:
        vmax = _vmax

    nrows = shape[1]
    ncols = shape[0]

    fig = _initialize_multiplot_figure(nrows, ncols)
    if plot_cbar is True:
        gs_main = gridspec.GridSpec(1, 2, fig, width_ratios=[20, 1])
    else:
        gs_main = gridspec.GridSpec(1, 1, fig)

    if title is not None:
        plt.suptitle(title)

    gs_sub = gridspec.GridSpecFromSubplotSpec(shape[1], shape[0], subplot_spec=gs_main[0])

    for i in range(len(data)):
        gs_indices = np.unravel_index(i, (shape[1], shape[0]))
        ax = fig.add_subplot(gs_sub[gs_indices], title=exp.sources[i].label)
        data[i] = np.reshape(data[i], exp.dim)
        data[i] = data[i][::-1, :]
        im = _display_img(data[i], ax, vmin, vmax, cmap, show_vals, hide_axes, sigdigs)

    if plot_cbar is True:
        cbar_ax = fig.add_subplot(gs_main[0, 1])
        fig.colorbar(im, cax=cbar_ax).set_label(label=cbar_label, size=18)

    plt.show()


def two_img_slider(set1, set2, labels1=None, labels2=None, cbar_label=None, spots_pos=None):
    """
    Plots two sets of images side by side with a single slider to select the index of images in the sets.

    Parameters
    ----------
    set1 : list
        First set of images.
    set2 : list
        Second set of images. Must be of the same length as set1.
    labels1 : str or list (optional)
        Either the same label for all images in the first set, or one label per image.
    labels2 : str or list (optional)
        Either the same label for all images in the first set, or one label per image.
    cbar_label : str (optional)
    spots_pos : array-like (optional)
        Positions to plot circles representing ion positions.
        Either the same set of positions for all images, or one set of positions for each pair of images.

    Returns
    -------
    ax1, ax2

    """
    if len(set1) != len(set2):
        raise ValueError('Number of images in each set must match.')

    rem_neg = lambda list: [x for x in list if x >= 0]
    rem_neg_2d = lambda list: [rem_neg(x) for x in list]

    if spots_pos is not None:
        spots_pos = np.array(spots_pos) - 0.5
        if len(spots_pos.shape) == 2:
            x_pos = rem_neg(spots_pos[:, 0].tolist())
            y_pos = rem_neg(spots_pos[:, 1].tolist())
            get_pos = lambda pos, j: pos
        elif len(spots_pos.shape) == 3:
            if spots_pos.shape[0] != len(set1):
                raise ValueError('Number of sets of positions must match number of images.')
            x_pos = rem_neg_2d(spots_pos[:, :, 0].tolist())
            y_pos = rem_neg_2d(spots_pos[:, :, 1].tolist())
            get_pos = lambda pos, j: pos[j]

    n_samples = len(set1)

    fig, (ax1, ax2) = plt.subplots(ncols=2, sharex=True, sharey=True)

    max1 = np.max(set1)
    max2 = np.max(set2)

    get_label = lambda label, j: label[j] if type(label) is list else label

    plt.subplots_adjust(bottom=0.20)
    i = 50
    ax1.set_title(get_label(labels1, i))
    ax2.set_title(get_label(labels2, i))
    im1 = ax1.matshow(set1[i], cmap='jet', vmin=0, vmax=max1)
    im2 = ax2.matshow(set2[i], cmap='jet', vmin=0, vmax=max2)
    if spots_pos is not None:
        get_circles = lambda x_pos, y_pos: [plt.Circle([x_pos[j], y_pos[j]], 0.9, fill=False, ec='r') for j in
                                            range(len(x_pos))]
        get_circlesX2 = lambda x_pos, y_pos: [get_circles(x_pos, y_pos), get_circles(x_pos, y_pos)]
        plt_circles = lambda ax, circles: [ax.add_artist(circles[j]) for j in range(len(circles))]
        circles = get_circlesX2(get_pos(x_pos, i), get_pos(y_pos, i))
        plt_circles(ax1, circles[0])
        plt_circles(ax2, circles[1])

    for ax, im in [[ax1, im1], [ax2, im2]]:
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, label=cbar_label)

    axcolor = 'lightgoldenrodyellow'
    ax_slider = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)

    slider = Slider(ax_slider, 'Source', 0, n_samples - 1, valinit=i, valstep=1)

    def update(val):
        i = int(slider.val)
        if spots_pos is not None:
            ax1.artists = []
            ax2.artists = []

            if len(get_pos(x_pos, i)) != len(get_pos(y_pos, i)):
                len1, len2 = len(get_pos(x_pos, i)), len(get_pos(y_pos, i))
                min_len = min(len1, len2)
                circles = get_circlesX2(get_pos(x_pos, i)[:min_len], get_pos(y_pos, i)[:min_len])
            else:
                circles = get_circlesX2(get_pos(x_pos, i), get_pos(y_pos, i))

            plt_circles(ax1, circles[0])
            plt_circles(ax2, circles[1])

        for ax, set, im, label in [[ax1, set1, im1, labels1], [ax2, set2, im2, labels2]]:
            im.set_data(set[i])
            ax.set_title(get_label(label, i))
        fig.canvas.draw_idle()

    slider.on_changed(update)

    fig.canvas.draw_idle()
    plt.show()

    return ax1, ax2


def plot_psf_list(psf_list, show_vals=False, cmap='viridis', hide_axes=True, plot_cbar=True, sigdigs=2, cbar_label=None,
                  vmin=None, vmax=None, title=None):
    """
    Plots a set of point spread functions.

    Parameters
    ----------
    psf_list : list of 2d arrays
    show_vals : bool (optional)
    cmap : str (optional)
    hide_axes : bool (optional)
    plot_cbar : bool (optional)
    sigdigs : int (optional)
        Number of significant digits to show if show_vals is True.
    cbar_label : str (optional)
    vmin : float (optional)
    vmax : float (optional)
    title : str (optional)

    Returns
    -------

    """

    if title is None:
        title = 'Point Spread Functions'

    # Plot given lists of data
    ncols = 2
    nrows = len(psf_list) // ncols + (len(psf_list) % ncols != 0)

    _vmin, _vmax = get_extrema(psf_list)
    if vmin is None:
        vmin = _vmin
    if vmax is None:
        vmax = _vmax

    fig = _initialize_multiplot_figure(nrows, ncols)
    if plot_cbar is True:
        gs_main = gridspec.GridSpec(1, 2, fig, width_ratios=[20, 1])
    else:
        gs_main = gridspec.GridSpec(1, 1, fig)

    plt.suptitle(title)

    gs_sub = gridspec.GridSpecFromSubplotSpec(nrows, ncols, subplot_spec=gs_main[0])

    for i in range(len(psf_list)):
        gs_indices = np.unravel_index(i, (nrows, ncols))
        ax = fig.add_subplot(gs_sub[gs_indices])
        # psf_list[i] = np.reshape(psf_list[i])
        psf_list[i] = psf_list[i][::-1, :]
        im = _display_img(psf_list[i], ax, vmin, vmax, cmap, show_vals, hide_axes, sigdigs)

    if plot_cbar is True:
        cbar_ax = fig.add_subplot(gs_main[0, 1])
    fig.colorbar(im, cax=cbar_ax).set_label(label=cbar_label, size=18)

    plt.show()


def bar_1d(data_sets, x_max, threshold=None, labels=None):
    """
    Plots a set of 1D bar graphs.

    Parameters
    ----------
    data_sets : list of array
    x_max : int
    threshold : int (optional)
        If specified, plots a dashed line at x=threshold
    labels : list of str (optional)

    Returns
    -------

    """
    fig = _initialize_multiplot_figure(1, len(data_sets))
    for i in range(len(data_sets)):
        data = data_sets[i]
        data = data[:x_max]

        ax = fig.add_subplot(1, len(data_sets), i + 1, xlabel='n')
        ax.bar(np.arange(x_max), data, align='edge')
        if labels is not None: ax.set_title(labels[i])
        if threshold is not None: ax.axvline(threshold, color='k', linestyle='--')
        ax.tick_params(axis='both', which='major', labelsize=10)
    plt.show()


def bar_2d(data_sets, x_max, y_max, colours=None, labels=None):
    """
    Plots a set of 2D bar graphs.

    Parameters
    ----------
    data_sets : list of ndarray
    x_max : int
    y_max : int
    colours : ndarray of str or tuples
        Colour of each bar. Dimensions of ndarray must match that of data_sets
    labels : list of str
        Length of list must match number of data sets.

    Returns
    -------

    """

    ncols = np.min([2, len(data_sets)])
    nrows = (len(data_sets) + 1) // 2

    fig = _initialize_multiplot_figure(nrows, ncols)
    for i in range(len(data_sets)):
        data = data_sets[i]
        data = data[:x_max, :y_max].ravel()
        # data2d = p_dists_plot[i]  # distribution to be plotted
        # data = np.transpose(data2d).ravel()  # flatten array

        _n1, _n2 = np.meshgrid(np.arange(x_max), np.arange(y_max))
        n1, n2 = _n1.ravel(), _n2.ravel()

        top = n1 + n2
        bottom = np.zeros_like(top)

        ax = fig.add_subplot(nrows, ncols, i + 1, projection='3d', xlabel='n1', ylabel='n2')
        ax.bar3d(n1, n2, bottom, 1, 1, data, color=colours, shade=True)
        if labels is not None: ax.set_title(labels[i])
        ax.tick_params(axis='both', which='major', labelsize=10)
    plt.show()


def _initialize_multiplot_figure(nrows, ncols):
    """
    Creates figure with specified number of subplots.

    Parameters
    ----------
    nrows : int
        Rows of subplots
    ncols : int
        Columns of subplots

    Returns
    -------
    matplotlib.Figure

    """
    widths = [6.4, 11, 15, 18, 20]
    heights = [4, 7, 9, 11, 11]
    fig = plt.figure(figsize=(widths[ncols - 1], heights[nrows - 1]))
    return fig


def _display_img(data, ax=None, vmin=None, vmax=None, cmap=None, show_vals=False, hide_axes=True, sigdigs=2):
    """
    Plots a single image.

    Parameters
    ----------
    data : ndarray
    ax : matplotlib.Axes (optional)
    vmin : float (optional)
    vmax : float (optional)
    cmap : str (optional)
    show_vals : bool (optional)
    hide_axes : bool (optional)
    sigdigs : int (optional)
        Number of significant digits to show if show_vals is True.

    Returns
    -------
    matplotlib.image.AxesImage

    """
    extent = [-1, 1, -1, 1]

    if ax is None:
        ax = plt.gca()
    im = ax.imshow(data, extent=extent, origin='lower', interpolation='None', vmin=vmin, vmax=vmax, cmap=cmap)
    if show_vals is True:
        show_values(data, extent, ax, sigdigs)
    if hide_axes is True:
        ax.axis('off')
    return im


def show_values(values, extent, ax, sigdigs):
    """
    Displays values of each pixel onto an image.

    Parameters
    ----------
    values : ndarray
    extent : list
        x_min, x_max, y_min, y_max
        Axes limits.
    ax : matplotlib.Axes
    sigdigs : int
    """

    dim = np.shape(np.transpose(values))
    [x_min, x_max, y_min, y_max] = extent
    jump_x = (x_max - x_min) / (2.0 * dim[0])
    jump_y = (y_max - y_min) / (2.0 * dim[1])
    x_positions = np.linspace(start=x_min, stop=x_max, num=dim[0], endpoint=False)
    y_positions = np.linspace(start=y_min, stop=y_max, num=dim[1], endpoint=False)
    str_format = '%0.' + str(sigdigs) + 'g'

    for y_index, y in enumerate(y_positions):
        for x_index, x in enumerate(x_positions):
            label = values[y_index, x_index]
            if values[y_index, x_index] > 10 ** (-sigdigs):
                str_format = '%0.' + str(sigdigs) + 'f'
            elif values[y_index, x_index] > 0:
                str_format = '%0.' + str(sigdigs) + 'e'
            label = str_format % label
            text_x = x + jump_x
            text_y = y + jump_y
            ax.text(text_x, text_y, label, color='black', ha='center', va='center', backgroundcolor='white')


def get_extrema(data):
    """
    Finds the minimum and maximum value of a set of images

    Parameters
    ----------
    data : List of ndarray

    Returns
    -------
    min_val : float
    max_val : float
    """
    d_array = np.array(data)
    print(d_array.dtype)
    print(d_array.shape)
    min_val = np.min([data[i] for i in range(len(data))])
    max_val = np.max([data[i] for i in range(len(data))])
    return min_val, max_val


def bar1D_broken_axis(y_vals, skip, x_vals=None, same_scale=True, height_ratios=None, **kwargs):
    """
    Creates a bar graph with a broken y axis.

    Parameters
    ----------
    y_vals : array-like
    skip : list
        [y1, y2] where y1 is the max val of the lower region, and y2 is the min val of the upper region
    x_vals : array-like (optional)
    same_scale : bool (optional)
    height_ratios : bool (optional)
        Ratio of height of top axis to bottom axis.
    kwargs : dict (optional)
        Passed into plt.bar

    Returns
    -------

    """

    y_vals = np.array(y_vals)
    if len(y_vals.shape) == 1:
        y_vals = np.array([y_vals])
        n_pts = len(y_vals)
        n_sets = 1
    else:
        n_sets, n_pts = y_vals.shape

    n_zeros = np.min(np.round(np.log10(np.max(y_vals))) - 2, 0)
    y_max = np.round((1.5 * np.max(y_vals) - 0.5 * skip[1]) / (10 ** n_zeros)) * (10 ** n_zeros)
    if height_ratios is None:
        height_ratios = [y_max - skip[1], skip[0]]
    if not same_scale:
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': height_ratios})
    else:
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    fig.subplots_adjust(hspace=0.05)  # adjust space between axes

    # plot the same data on both axes
    for i in range(n_sets):
        if x_vals is None:
            ax1.bar(range(n_pts), y_vals[i, :], **kwargs)
            ax2.bar(range(n_pts), y_vals[i, :], **kwargs)
        else:
            ax1.bar(x_vals, y_vals[i, :], **kwargs)
            ax2.bar(x_vals, y_vals[i, :], **kwargs)

    ax1.set_ylim(bottom=skip[1], top=y_max)
    ax2.set_ylim(top=skip[0])

    # hide the spines between ax and ax2
    ax1.spines['bottom'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax1.tick_params(labeltop=False)  # don't put tick labels at the top
    ax1.set_xticks([])
    ax2.xaxis.tick_bottom()

    d = 0.5  # proportion of vertical to horizontal extent of the slanted line
    kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
                  linestyle="none", color='k', mec='k', mew=1, clip_on=False)
    ax1.plot_imgs([0, 1], [0, 0], transform=ax1.transAxes, **kwargs)
    ax2.plot_imgs([0, 1], [1, 1], transform=ax2.transAxes, **kwargs)

    plt.show()


def plot_spots(ax, x_pos, y_pos, radius, color='r'):
    """
    Plots a series of hollow circles.

    Parameters
    ----------
    ax : matplotlib.Axes
    x_pos : list
    y_pos : list
    radius : float
    color : str

    Returns
    -------

    """

    n_ions = len(x_pos)
    if len(color) == 1:
        color = color * n_ions
    for i in range(n_ions):
        ax.add_artist(plt.Circle((x_pos[i], y_pos[i]), radius, fill=False, color=color[i]))
