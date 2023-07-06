import numpy as np
import IonImaging.numpy_sim as IonImaging

run = IonImaging.NumpySimulation([3, 1], tau=850, s=0.7, na=0.4, qe=0.1, r_bg=1)
run.add_ion(state=True, psf=[0.5,0.3,0.2]) #[0.29, 0.24, 0.19, 0.14, 0.09, 0.05]
run.add_bkg()
mc_results = run.mc_sim(10000)
avgs_mc = np.average(mc_results, axis=0).flatten()
dist_analytic = run.anal_sim([0, 1, 2])
avgs_analytic = []
n_axes = len(dist_analytic.shape)
for i in range(n_axes):
    sum_axes = list(range(n_axes))
    sum_axes.pop(i) #the sailor man
    sum_axes = tuple(sum_axes)  # tuple of indices from 0 to n_axes without i
    avgs_analytic.append(np.sum(np.arange(np.shape(dist_analytic)[i]) * np.sum(dist_analytic, axis=sum_axes)))
avgs_analytic = np.array(avgs_analytic).flatten()
residual = np.sum((avgs_mc - avgs_analytic) ** 2)