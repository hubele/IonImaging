"""Functions to assist in measuring state and calculating measurement fidelity."""
import numpy as np
import matplotlib.cm as cm


def measure_f_discrete(states, dists, measured_states):
    """
    Calculates the fidelity of each state based on the probability of generating a set of photon counts from ions in
    each state, and from the predicted state from each set of photon counts.

    Parameters
    ----------
    states : list
        All possible states
    dists : list of ndarrays
        Number of elements in list must match the length of states. First element is probability of generating a
        certain set of photon counts in each pixel from ions in the first state; second element corresponds to second
        state, etc.
    measured_states : ndarray
        The predicted state for each set of photon counts (n1,n2,...).

    Returns
    -------
    list
        Fidelity of measuring each state, in the same order as provided in "states".

    """

    f = []
    for i in range(len(states)):
        state = states[i]
        dist = dists[i]

        f.append(np.sum(dist[measured_states == state]))
    return f


def measure_f_rand(dists, state_probs):
    """
    Calculates the fidelity of each state based on the probability of generating a set of photon counts from ions in
    each state, and from the probability of predicting a certain state from each set of photon counts.

    Parameters
    ----------
    dists : list of ndarrays
        First element is probability of generating a certain set of photon counts (n1,n2,...) in each pixel from ions in
        the first state; second element corresponds to second state, etc.
    state_probs : list
        Number of elements in list must match length of list "dists". First element is probability of measuring the
        first state for each set of photon counts (n1,n2,...); second element corresponds to second state, etc..

    Returns
    -------
    list
        Fidelity of measuring each state, in the same order as provided in "dists".

    """
    f = []
    for i in range(len(dists)):
        f.append(np.sum(dists[i] * state_probs[i]))
    return f


def assign_colours_discrete(states):
    """
    Assigns a unique color to each possible state, for plotting.

    Parameters
    ----------
    states : ndarray

    Returns
    -------

    """
    cmap = cm.get_cmap('Set1')
    colour_list = [cmap(i) for i in range(cmap.N)]
    states = np.array(states)
    shape = states.shape
    states = states.flatten()
    assigned_colours = np.empty(states.size, dtype=tuple)
    unique_states = []
    for i in range(len(states)):
        if states[i] not in unique_states:
            unique_states.append(states[i])
        assigned_colours[i] = colour_list[unique_states.index(states[i])]
    assigned_colours = assigned_colours.reshape(shape)
    return assigned_colours


def assign_colours_rand(states_probs):
    """
    Only works for a 2-state system. Compares the probability for each state to get a certain photon count P(n1,n2) and
    assigns the color based on the ratio of the probabilities that the ion will be measured as bright/dark.

    Parameters
    ----------
    states_probs : list

    Returns
    -------

    """

    cmap = cm.get_cmap('PiYG')
    states_probs = np.array(states_probs[0])
    shape = states_probs.shape
    states_probs = states_probs.flatten()
    assigned_colours = np.empty(states_probs.size, dtype=tuple)
    for i in range(states_probs.size):
        assigned_colours[i] = cmap(states_probs[i])
    assigned_colours = assigned_colours.reshape(shape)
    return assigned_colours
