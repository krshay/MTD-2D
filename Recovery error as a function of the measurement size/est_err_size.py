# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 14:27:49 2020

@author: Shay Kreymer
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import multiprocessing as mp
from Utils.calc_err_size import calc_err_size_knownpsftsf, calc_err_size_Algorithm1, calc_err_size_nopsftsf

plt.close("all")

if __name__ == '__main__':
    # Code to reproduce Fig. 5 in the paper. 
    # Estimation error as a function of measurement size for 3 cases: known pair separation functions, Algorithm 1, and assuming the well-separated model.
    # %% Preliminary definitions
    N = 30000
    Niters = 100
    L = 5
    ne = 10
    Nsizes = 15
    sizes = np.logspace(np.log10(1000), np.log10(N), Nsizes).astype(int)
    
    num_cpus = mp.cpu_count()
    # %% Known psf and tsf
    pool = mp.Pool(num_cpus)
    Sknown = pool.starmap(calc_err_size_knownpsftsf, [[L, ne, sizes, i] for i in range(Niters)])
    pool.close()
    pool.join() 
    
    # %% Algorithm1
    pool = mp.Pool(num_cpus)
    SAlgorithm1 = pool.starmap(calc_err_size_Algorithm1, [[L, ne, sizes, i] for i in range(Niters)])
    pool.close()
    pool.join()
    
    # %% No psf and tsf
    sizes_no = np.logspace(np.log10(1000), np.log10(N), 5).astype(int)
    pool = mp.Pool(num_cpus)
    Sno = pool.starmap(calc_err_size_nopsftsf, [[L, ne, sizes, i] for i in range(Niters)])
    pool.close()
    pool.join()
    
    # %% Calculations
    errsAlgorithm1 = np.zeros((Niters, Nsizes))

    for j in range(Niters):
        errsAlgorithm1[j, :] = SAlgorithm1[j][0][np.arange(Nsizes), np.argmin(SAlgorithm1[j][1], axis=1)]
    errsAlgorithm1_median = stats.trim_mean(errsAlgorithm1, 0.06, 0)
    
    errsknown = np.zeros((Niters, Nsizes))

    for j in range(Niters):
        errsknown[j, :] = Sknown[j][0][np.arange(Nsizes), np.argmin(Sknown[j][1], axis=1)]
    errsknown_median = stats.trim_mean(errsknown, 0.06, 0)
    
    errs_no = np.zeros((Niters, Nsizes))

    for j in range(Niters):
        errs_no[j, :] = Sno[j][0][np.arange(Nsizes), np.argmin(Sno[j][1], axis=1)]
    errs_no_median = stats.trim_mean(errs_no, 0.06, 0)
    
    # %% plots
    plt.close("all")
    with plt.style.context('ieee'):
        fig = plt.figure()
        plt.loglog(sizes**2, errsknown_median[3]*(sizes**2/sizes[3]**2)**(-1/2), 'k--', label='_nolegend_', lw=0.5)
        plt.loglog(sizes**2, errsknown_median, '.-b', label=r'known $\xi$ and $\zeta$')
        plt.loglog(sizes**2, errsAlgorithm1_median[3]*(sizes**2/sizes[3]**2)**(-1/2), 'k--', label='_nolegend_', lw=0.5)
        plt.loglog(sizes**2, errsAlgorithm1_median, '.--r', label='Algorithm 1')
        plt.loglog(sizes_no**2, errs_no_median, ':g', label=r'no $\xi$ and $\zeta$')
        plt.legend(loc=(0.5, 0.62))
        plt.xlabel('Measurement size [pixels]')
        plt.ylabel('Mean estimation error')
        fig.tight_layout()
        plt.show()
        