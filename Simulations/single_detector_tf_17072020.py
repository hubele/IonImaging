# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 14:38:26 2020

@author: Scott
"""
import analytic_sim
import classes
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf
import tensorflow_probability as tfp
import IonImaging

tfd = tfp.distributions
tfb = tfp.bijectors


# %%
def efficiency(na):
    """Collection efficiency for a given numerical aperture."""
    return 0.5 * (1 - np.cos(np.arcsin(na)))


# %%
s = 0.5  # I/I_sat
tau_D = 500e-6  # Detection time (s)
qe = 0.2  # Quantum efficiency
na = 0.4  # Numerical aperture
r_bg = 1000  # Background rate (counts/s)

gamma = 20e6
delta = 0
Delta_1 = 14.7e9  # HFS + HFP
Delta_2 = 2.1e9  # HFP

tau_L1 = 36 * (Delta_1 ** 2) / (s * (gamma ** 3))
tau_L2 = 36 * (Delta_2 ** 2) / (s * (gamma ** 3))

eta = qe * efficiency(na)  # total collection efficiency
alpha_1 = (2 / 9) * (1 + s + (2 * delta / gamma) ** 2) * (gamma / 2 / Delta_1) ** 2
alpha_2 = (2 / 9) * (1 + s + (2 * delta / gamma) ** 2) * (gamma / 2 / Delta_2) ** 2
alpha_1_over_eta = alpha_1 / eta
alpha_2_over_eta = alpha_2 / eta
lambda_0 = tau_D * eta * s * (gamma / 2) / (1 + s + (2 * delta / gamma) ** 2)

# %%
dark_state = tfd.JointDistributionNamed(dict(
    tau_decay=tfd.Exponential(1 / tau_L1),
    dark_time=lambda tau_decay: tfd.Deterministic(tf.math.minimum(tau_decay, tau_D)),
    n_avg=lambda dark_time: tfd.Deterministic((1 - dark_time / tau_D) * lambda_0),
    n_ion=lambda n_avg: tfd.Poisson(n_avg),
    n_bkg=tfd.Poisson(r_bg * tau_D),
    n_total=lambda n_ion, n_bkg: tfd.Deterministic(n_ion + n_bkg)
))

bright_state = tfd.JointDistributionNamed(dict(
    tau_decay=tfd.Exponential(1 / tau_L2),
    bright_time=lambda tau_decay: tfd.Deterministic(tf.math.minimum(tau_decay, tau_D)),
    n_avg=lambda bright_time: tfd.Deterministic((bright_time / tau_D) * lambda_0),
    n_ion=lambda n_avg: tfd.Poisson(n_avg),
    n_bkg=tfd.Poisson(r_bg * tau_D),
    n_total=lambda n_ion, n_bkg: tfd.Deterministic(n_ion + n_bkg)
))

n_trials = 500000
sess = tf.Session()
dark_counts = sess.run(dark_state.sample(n_trials)['n_total'])
bright_counts = sess.run(bright_state.sample(n_trials)['n_total'])

p_bright_anal, p_dark_anal, p_bg_anal = analytic_sim.calc_dists(tau_D * 1e6, s, na, qe, r_bg / 1000)

p_bright_bg = IonImaging.misc.convolve(p_bright_anal, p_bg_anal)
p_dark_bg = IonImaging.misc.convolve(p_dark_anal, p_bg_anal)

n_max = 10
plt.figure()
ax = plt.axes(title='Prepared Dark', xlabel='Counts', ylabel='Probability')
plt.hist(dark_counts, density=True, bins=range(n_max))
plt.plot(np.arange(n_max) + 0.5, p_dark_bg[:n_max], 'ko')
plt.tight_layout()

n_max = 30
plt.figure()
ax = plt.axes(title='Prepared Bright', xlabel='Counts', ylabel='Probability')
plt.hist(bright_counts, density=True, bins=range(n_max))
plt.plot(np.arange(n_max) + 0.5, p_bright_bg[:n_max], 'ko')
plt.tight_layout()
#
# sess.close()
