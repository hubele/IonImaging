"""
Created on Sun Aug 30 13:40 2020

@author: Scott Hubele
"""
from IonImaging import tfp_sampling

# %% Parameters

imaging_params = dict(
    tau=500,  # μs
    s=0.5,  # I/I_sat
    na=0.4,
    qe=0.2,
    r_bg=1 / 3.,  # counts/ms/pixel
    angle=0.,
    spacing=1.8,
    spot_diam=1.8
)

sampling_params = dict(
    n_images=5000,
    n_bunch=1,
    n_testing=50,
    plot_samples=False,
    save_data=False,
    verbose=1,
    label='20Ions_RandomPos'
)

# %%
run = tfp_sampling.TFPSimulation([20, 20], **imaging_params)

n_ions = 10
for _ in range(n_ions):
    run.add_ion()
run.add_bkg()

run.randomize_ion_pos()