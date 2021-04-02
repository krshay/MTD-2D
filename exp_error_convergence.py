# -*- coding: utf-8 -*-
"""
Created on Sun Feb 21 19:01:56 2021

@author: kreym
"""

import numpy as np
import matplotlib.pyplot as plt

import scipy.stats as stats


# %% loads
errs_approx = np.load('Results/Error with micrograph size/errs_approx.npy')

errs_true = np.load('Results/Error with micrograph size/errs_true.npy')

errs_no = np.load('Results/Error with micrograph size/errs_no_0_50.npy')

sizes = np.load('Results/Error with micrograph size/sizes.npy')

errs_approx = errs_approx[ :100, :]

errs_true = errs_true[ :100, :]

errs_no = errs_no[ :100, :]

# bool_idx = np.zeros(100, dtype=bool)
# bool_idx[[26, 36, 39, 55, 75]] = True

# errs_true_reduced = errs_true[~bool_idx, :]

# errs_approx_reduced = errs_approx[~bool_idx, :]

errs_approx_mean = np.median(errs_approx, 0)

errs_true_mean = np.median(errs_true, 0)

errs_no_mean = np.median(errs_no, 0)

# errs_approx_mean = stats.trim_mean(errs_approx, 0.05, 0)

# errs_true_mean = stats.trim_mean(errs_true, 0.05, 0)

# errs_no_mean = stats.trim_mean(errs_no, 0.05, 0)

# %% plots
plt.close("all")
with plt.style.context('ieee'):
    fig = plt.figure()
    
    plt.loglog(sizes**2, errs_true_mean[0]*(sizes**2/sizes[0]**2)**(-1/2), 'k--', label='_nolegend_', lw=0.5)
    plt.loglog(sizes**2, errs_true_mean, '.-b', label=r'known $\xi$ and $\zeta$')

    plt.loglog(sizes**2, errs_approx_mean[0]*(sizes**2/sizes[0]**2)**(-1/2), 'k--', label='_nolegend_', lw=0.5)
    plt.loglog(sizes**2, errs_approx_mean, '.--r', label='Algorithm 1')
    
    plt.loglog(sizes**2, errs_no_mean, ':g', label=r'no $\xi$ and $\zeta$')

    plt.legend(loc=(0.5, 0.55))#, fontsize=6)
    
    plt.xlabel('Measurement size [pixels]')
    
    plt.ylabel('Median estimation error')
    fig.tight_layout()
    plt.show()

    
    plt.savefig(r'C:\Users\kreym\Google Drive\Thesis\Documents\Article\figures\error_convergence_experiment.pdf')

# plt.close("all")