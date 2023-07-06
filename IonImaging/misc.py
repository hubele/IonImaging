import numpy as np
from scipy.signal import convolve as scipy_convolve


def convolve(*arrays):
    """
    Convolves an arbitrary number of arrays together.

    Parameters
    ----------
    arrays : list of numPy arrays
        Arrays to be convolved together.

    Returns
    -------
    ndarray
        Convolution of all arrays.

    """

    output = np.ones(shape=tuple([1] * np.size(np.shape(arrays[0]))))
    for array in arrays:
        output = scipy_convolve(output, array)
    return output


def index_array(i, array):
    """
    Returns the i-th value of a 1D array.

    Used to define vectorized version of the function which creates an defines an array indexed from a  array using a
    list of indices.

    Parameters
    ----------
    i : int
        Index of array.
    array : list

    Returns
    -------
    i-th value of given array if i is in bounds. Returns 0 if i is greater than list size.

    """

    if i >= np.size(array):
        return 0
    else:
        return array[i]


array_by_indices = np.vectorize(index_array, excluded=[1])


def decimal_to_binary(*dec_num, out_type=str, match_len=True, n_bits=None, get_digit=None):
    """
    Converts decimal inputs to binary

    Parameters
    ----------
    dec_num : int
        Decimal number(s) to convert
    out_type : type
    match_len : bool
    n_bits : int
        Number of bits in binary output. Must be greater than minimum digits required to express binary number. If
        specified, overrides match_len.
    get_digit : int
        If specified, overrides out_type and match_len. Returns only the i-th digit of the binary numbers.

    Returns
    -------
    list

    """
    binary_nums = []
    if get_digit is not None:
        out_type = str
        match_len = True

    for n in dec_num:
        binary_nums.append('')
        while n >= 1:
            binary_nums[-1] = str(n % 2) + binary_nums[-1]
            n = n // 2

    if (match_len is True) and (dec_num != (0,)):
        max_len = len(decimal_to_binary(np.max(dec_num), match_len=False))
    if n_bits is not None:
        max_len = n_bits

    for i in range(len(binary_nums)):
        binary_nums[i] = out_type(binary_nums[i])
        if out_type is str and match_len is True:
            leading_zeros = '0' * (max_len - len(binary_nums[i]))
            binary_nums[i] = leading_zeros + binary_nums[i]

    if get_digit is not None:
        for i in range(len(binary_nums)):
            binary_nums[i] = binary_nums[i][get_digit]

    if len(dec_num) == 1:
        binary_nums = binary_nums[0]

    return binary_nums


def shuffle_together(*arrays):
    """
    Shuffles arrays in the same order.

    Parameters
    ----------
    arrays : list of ndarray

    Returns
    -------

    """
    length = arrays[0].shape[0]
    shuffled_arrays = []
    indices = np.arange(length, dtype=int)
    np.random.shuffle(indices)

    for array in arrays:
        shuffled_arrays.append(array[indices])

    return shuffled_arrays


def efficiency(na):
    """
    Collection efficiency for a given numerical aperture.

    Parameters
    ----------
    na : float

    Returns
    -------
    float
        Collection efficiency.

    """

    return 0.5 * (1 - np.cos(np.arcsin(na)))


def gen_random_state(n_bits, out_fmt='bin_str', n_samples=1, allow_duplicates=True, **kwargs):
    """
    Generates a list of randomly generated ion states

    Parameters
    ----------
    n_samples : int
    n_bits : int
    out_fmt : 'bin_str' | 'bin_array' | 'dec'
    allow_duplicates : bool
        Allows whether or not list of output states. If true, n_samples must be less than the number of possible states.

    Returns
    -------
    list

    """

    state = []
    for _ in range(n_samples):
        state.append(''.join(np.random.choice([0, 1], n_bits).astype('str').tolist()))

    if allow_duplicates is False:
        if n_samples > int(2**n_bits):
            raise ValueError('Number of unique states cannot exceed number of possible states.')

        duplicates_might_exist = True
        while duplicates_might_exist:
            state = remove_duplicates(state)
            n_removed = n_samples - len(state)
            if n_removed == 0:
                duplicates_might_exist = False
            else:
                new_states = gen_random_state(n_bits, out_fmt, n_removed, allow_duplicates)
                for new_state in new_states:
                    state.append(new_state)

    state = set_state_fmt(state)

    return state


def remove_duplicates(array):
    """
    Removes duplicate entries in an array.

    Parameters
    ----------
    array : array-like

    Returns
    -------
    list

    """
    new_array = np.copy(array)
    for element in array:
        indices = np.where(new_array == element)[0]
        if indices.size > 1:
            new_array = np.delete(new_array, indices[1:])
    return new_array.tolist()


def as_bin_array(input_nums, out_type='int', n_bits=None):
    """
    Converts binary numbers to array format, either int or float.

    Parameters
    ----------
    input_nums : list
        list elements are binary numbers, either as str, int, or array of floats or ints
    out_type : data type (optional)
        type of output array
    n_bits : int (optional)
        Only has an effect when input_num is a decimal number

    Returns
    -------
    ndarray
        Zeroth axis has same length as list passed in, first axis has length equal to the number of bits

    """

    n_nums = len(input_nums)
    if type(input_nums[0]) is str or type(input_nums[0]) is np.str_:
        n_bits = len(input_nums[0])
        binary_out = np.zeros((n_nums, n_bits), dtype=out_type)
        for n in range(n_nums):
            for i in range(n_bits):
                binary_out[n, i] = input_nums[n][i]

    elif type(input_nums[0]) is int or type(input_nums[0]) is np.int_:
        if n_bits is None:
            bin_nums_str = decimal_to_binary(*input_nums, out_type=str)
        else:
            bin_nums_str = decimal_to_binary(*input_nums, n_bits=n_bits, out_type=str)
        binary_out = as_bin_array(bin_nums_str, out_type=out_type)

    else:  # assume array-like of floats or ints
        indv_bits = [np.array(input_nums[n]) for n in range(n_nums)]
        n_bits = indv_bits[0].size
        binary_out = np.zeros((n_nums, n_bits), dtype=out_type)

        for n in range(n_nums):
            for i in range(n_bits):
                binary_out[n, i] = input_nums[n][i]

    return binary_out


def as_bin_str(input_nums, n_bits=None):
    """
    Converts binary numbers to str format, either int or float.

    Parameters
    ----------
    input_nums : list
        list elements are binary numbers, either as str, int (decimal), or array of floats or ints
    n_bits : int (optional)
        Only has an effect when input_num is a list of int

    Returns
    -------
    list
        All binary numbers, each as a str

    """
    bin_array = as_bin_array(input_nums, out_type='int', n_bits=n_bits)
    n_nums, n_bits = bin_array.shape

    binary_out = []
    for n in range(n_nums):
        binary_out.append('')
        for i in range(n_bits):
            binary_out[-1] = binary_out[-1] + str(bin_array[n, i])

    return binary_out


def as_dec_array(input_nums, out_type='int'):
    """
    Converts binary numbers to an array of decimal numbers, either int or float.

    Parameters
    ----------
    input_nums : list
        list elements are binary numbers, either as str, int (decimal), or array of floats or ints
    out_type : 'int' | 'float'
        dtype of output array

    Returns
    -------
    ndarray
        1D array, the decimal equivalent of the passed in values

    """
    if type(input_nums[0]) is int or type(input_nums[0]) is np.int_:
        decimal_out = np.array(input_nums)

    else:
        bin_array = as_bin_array(input_nums, out_type='int')
        n_nums, n_bits = bin_array.shape

        decimal_out = np.zeros(n_nums, dtype=out_type)
        for n in range(n_nums):
            for i in range(n_bits):
                decimal_out[n] = decimal_out[n] + (2 ** (n_bits - i - 1)) * bin_array[n, i]

    return decimal_out


def set_state_fmt(input_nums, as_type='bin_str', **kwargs):
    """
    Converts between forms of representing ion states.

    Parameters
    ----------
    input_nums : list
    as_type : str
        'bin_array' or 'bin_str'
    kwargs
        Passed to function corresponding to as_type

    Returns
    -------
    out_states : list

    """
    if as_type is 'bin_str':
        out_states = as_bin_str(input_nums, **kwargs)
    elif as_type is 'bin_array':
        out_states = as_bin_array(input_nums, **kwargs)
    else:
        raise ValueError('Invalid output type specified.')

    return out_states


def is_in_circle(xy, origin, radius):
    """
    Determines if a point is located within a circle.

    Parameters
    ----------
    xy : list
        Point (x,y)
    origin : list
        Center of circle (x,y)
    radius : float

    Returns
    -------
    0 or 1
        0 if not in circle, 1 if in circle.

    """
    ox = origin[0]
    oy = origin[1]

    x = xy[0]
    y = xy[1]

    dist_to_origin = np.sqrt((x - ox) ** 2 + (y - oy) ** 2)

    if dist_to_origin >= radius:
        return 0
    else:
        return 1


def gen_rect_mesh(shape, extent, on_border=False):
    """
    Generates a grid of points within a specified rectangle

    Parameters
    ----------
    shape : tuple
        width, height
        Number of points in mesh in each direction
    extent : list
        [x_min, x_max, y_min, y_max]
        Coordinates of border of region to distribute points
        Points are not generated on the border by default
    on_border : bool
        Whether points may be placed on the border

    Returns
    -------
    list
        point coordinates expressed as [x,y]

    """
    if on_border:
        x_pts = np.linspace(extent[0], extent[1], shape[1], endpoint=True)
        y_pts = np.linspace(extent[2], extent[3], shape[0], endpoint=True)
    else:
        x_pts = np.linspace(extent[0], extent[1], shape[1], endpoint=False)
        y_pts = np.linspace(extent[2], extent[3], shape[0], endpoint=False)

        x_pts = x_pts + (x_pts[1] - x_pts[0]) / 2
        y_pts = y_pts + (y_pts[1] - y_pts[0]) / 2

    coords_x, coords_y = np.meshgrid(x_pts, y_pts)
    coords = np.array([coords_x, coords_y]).transpose().flatten()
    coords = coords.reshape(coords.size // 2, 2).tolist()

    return coords


def crop_images(imgs):
    """
    Removes columns or rows of a set of images where elements are zero for every given image.

    Parameters
    ----------
    imgs : array-like

    Returns
    -------
    ndarray

    """

    imgs = np.array(imgs)
    if imgs.min() < 0:
        raise ValueError('Images cannot have negative values.')

    non_empty_rows = imgs.sum((0, 2)) == 0
    non_empty_cols = imgs.sum((0, 1)) == 0
    imgs_cropped = imgs[:, non_empty_rows, non_empty_cols]

    return imgs_cropped