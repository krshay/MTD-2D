# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 14:37:57 2021

@author: Shay Kreymer
"""
# %% imports
import numpy as np
import matplotlib.pyplot as plt

import scipy

from Utils.fb_funcs import expand_fb, calcT
from Utils.generate_clean_micrograph_2d import generate_clean_micrograph_2d_rots
from Utils.funcs_calc_moments import M2_2d, M3_2d
from Utils.psf_tsf_funcs import full_psf_2d
from Utils.psf_tsf_funcs import full_tsf_2d
import Utils.optimization_funcs_rot
from Utils.psf_tsf_funcs import makeExtraMat
from Utils.psf_tsf_funcs import maketsfMat

# %% loads
errs_approx_009 = np.load('../Results/gamma_exp/errs_approx_009.npy')
errs_approx_009 = errs_approx_009 / 100
errs_true_009 = np.load('../Results/gamma_exp/errs_true_009.npy')
errs_true_009 = errs_true_009 / 100
errs_well_separated_009 = np.load('../Results/gamma_exp/errs_well_separated_009.npy')
errs_well_separated_009 = errs_well_separated_009 / 100
history_approx_009 = np.load('../Results/gamma_exp/history_approx_009.npy')
history_true_009 = np.load('../Results/gamma_exp/history_true_009.npy')
history_well_separated_009 = np.load('../Results/gamma_exp/history_well_separated_009.npy')

# %% plots
with plt.style.context('ieee'):
    plt.close("all")
    iters = list(range(101))
    fig = plt.figure()
    plt.plot(iters, 0.10 * np.ones((101, )), label='_nolegend_')
    l1 = plt.plot(iters, history_true_009)
    l2 = plt.plot(iters, history_approx_009)
    l3 = plt.plot(iters, history_well_separated_009)


    plt.xticks(list(range(0,101,10)))

    
    plt.xlabel('iterations')
    
    plt.ylabel('$\gamma$')

    plt.grid(True, which='both', axis='both')
    
    plt.xlim(0, 100)
    plt.ylim(0.09, 0.11)

    labels = [r"known $\xi$ and $\zeta$", r"approximated $\xi$ and $\zeta$", r"no $\xi$ and $\zeta$"]#, r"accurate $\xi$ and $\zeta$; $\gamma_{\mathrm{init}} = 0.11$", r"approximated $\xi$ and $\zeta$; $\gamma_{\mathrm{init}} = 0.11$"]
    plt.legend(labels, loc=1, fontsize=7)

    fig.tight_layout()
    plt.show()
    
    plt.savefig(r'C:\Users\kreym\Google Drive\Thesis\Documents\Article\figures\gamma_experiment.pdf')
