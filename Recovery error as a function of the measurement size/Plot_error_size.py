# -*- coding: utf-8 -*-
"""
Created on Sun Feb 21 19:01:56 2021

@author: kreym
"""

import numpy as np
import matplotlib.pyplot as plt

# Data produced using est_err_size_arbitrary_spacing_distribution_accuratepsftsf.py, est_err_size_arbitrary_spacing_distribution_approximatepsftsf.py and est_err_size_arbitrary_spacing_distribution_nopsftsf.py

# %% loads
errs_approx = np.load('../Results/Error with micrograph size/errs_approx.npy')

errs_true = np.load('../Results/Error with micrograph size/errs_true.npy')

errs_no = np.load('../Results/Error with micrograph size/errs_no_0_50.npy')

sizes = np.load('../Results/Error with micrograph size/sizes.npy')

# %% Calculations
errs_approx_median = np.median(errs_approx, 0)

errs_true_median = np.median(errs_true, 0)

errs_no_median = np.median(errs_no, 0)

# %% plots
plt.close("all")
with plt.style.context('ieee'):
    fig = plt.figure()
    
    plt.loglog(sizes**2, errs_true_median[0]*(sizes**2/sizes[0]**2)**(-1/2), 'k--', label='_nolegend_', lw=0.5)
    plt.loglog(sizes**2, errs_true_median, '.-b', label=r'known $\xi$ and $\zeta$')

    plt.loglog(sizes**2, errs_approx_median[0]*(sizes**2/sizes[0]**2)**(-1/2), 'k--', label='_nolegend_', lw=0.5)
    plt.loglog(sizes**2, errs_approx_median, '.--r', label='Algorithm 1')
    
    plt.loglog(sizes**2, errs_no_median, ':g', label=r'no $\xi$ and $\zeta$')

    plt.legend(loc=(0.5, 0.55))#, fontsize=6)
    
    plt.xlabel('Measurement size [pixels]')
    
    plt.ylabel('Median estimation error')
    fig.tight_layout()
    plt.show()
    
    plt.savefig(r'C:\Users\kreym\Google Drive\Thesis\Documents\Article\figures\error_convergence_experiment.pdf')
