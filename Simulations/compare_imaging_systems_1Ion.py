# -*- coding: utf-8 -*-
"""
Created on Thu Jul 06 20:47 2020

@author: Scott Hubele
"""
import IonImaging.numpy_sim as Imaging
import numpy as np

# %% Parameters
tau_d = 800  # 400
s = 0.5
na = 0.3
qe = 0.2

# %% Set up imaging systems
run1 = Imaging.NumpySimulation([1, 1], tau=tau_d, s=s, na=na, qe=2 * qe, r_bg=2)
ion = run1.add_ion(state=0, psf=[1.], label='Ion 1')
bkg = run1.add_bkg(label='Background')
run1.plot_psfs(shape=(2, 1), show_vals=True)

run2 = Imaging.NumpySimulation([2, 1], tau=tau_d, s=s, na=na, qe=2 * qe, r_bg=2)
ion = run2.add_ion(state=1, psf=[0.5, 0.5], label='Ion 1')
bkg = run2.add_bkg(label='Background')
dist_anal = run2.anal_sim([0, 1])
run2.plot_psfs(shape=(2, 1), show_vals=True)

# %% Set methods for judging fidelity
methods_1d = ['thresh_1d']
labels_1d = ['Single CCD - 1 Pixel' + label for label in ['(Threshold)']]
methods_2d = ['thresh_5050', 'thresh_diag', 'compp_disc']
labels_2d = ['Single CCD - 2 Pixels' + label for label in
             ['(Threshold 5050)', '(Diagonal Threshold)', '(Compare P(n1,n2))']]

# %% Run tests
kwargs1 = [{'n_max': 20}]
for i in range(len(methods_1d)):
    f_d, f_b = run1.measure_fidelity([0], method=methods_1d[i], **kwargs1[i])
    f_avg = (f_d + f_b) / 2
    print(labels_1d[i])
    print('F_d = %0.6f' % f_d)
    print('F_b = %0.6f' % f_b)
    print('F_avg = %0.6f' % f_avg)
    print('')

kwargs2 = [dict(n1_max=15, n2_max=15), dict(weights=[1, 1], n1_max=15, n2_max=15), dict(n1_max=15, n2_max=15)]
for i in range(len(methods_2d)):
    f_d, f_b = run2.measure_fidelity([0, 1], method=methods_2d[i], **kwargs2[i])
    f_avg = (f_d + f_b) / 2
    print(labels_2d[i])
    print('F_d = %0.4f' % f_d)
    print('F_b = %0.4f' % f_b)
    print('F_avg = %0.4f' % f_avg)
    print('')
