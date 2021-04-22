# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 14:27:49 2020

@author: Shay Kreymer
"""

import numpy as np
import matplotlib.pyplot as plt

import multiprocessing as mp

from Utils.calc_err_size import calc_err_size_knownpsftsf, calc_err_size_Algorithm1, calc_err_size_nopsftsf


plt.close("all")

if __name__ == '__main__':
    N = 30000
    Niters = 100
    L = 5
    ne = 10
    Nsizes = 15
    sizes = np.logspace(np.log10(1000), np.log10(N), Nsizes).astype(int)
    
    num_cpus = mp.cpu_count()
    # %% Known psf and tsf
    pool = mp.Pool(num_cpus)
    Sknown = pool.starmap(calc_err_size_knownpsftsf, [[L, ne, N, sizes, i+25] for i in range(Niters)])
    pool.close()
    pool.join() 
    
    np.save('Sknown.npy', np.array(Sknown))
    
    # %% Algorithm1
    pool = mp.Pool(num_cpus)
    SAlgorithm1 = pool.starmap(calc_err_size_Algorithm1, [[L, ne, N, sizes, i+25] for i in range(Niters)])
    pool.close()
    pool.join()
    
    np.save('SAlgorithm1.npy', np.array(SAlgorithm1))
    
    # %% No psf and tsf
    pool = mp.Pool(num_cpus)
    Sno = pool.starmap(calc_err_size_nopsftsf, [[L, ne, N, sizes, i] for i in range(Niters)])
    pool.close()
    pool.join()
    
    np.save('Sno.npy', np.array(Sno))
    
    # %% Calculations

    errsAlgorithm1 = np.zeros((Niters, Nsizes))

    for j in range(Niters):
        errsAlgorithm1[j, :] = SAlgorithm1[j][0][np.arange(Nsizes), np.argmin(SAlgorithm1[j][1], axis=1)]
    errsAlgorithm1_median = np.median(errsAlgorithm1, 0)
    
    errsknown = np.zeros((Niters, Nsizes))

    for j in range(Niters):
        errsknown[j, :] = Sknown[j][0][np.arange(Nsizes), np.argmin(Sknown[j][1], axis=1)]
    errsknown_median = np.median(errsknown, 0)
    
    # %% plots
    plt.close("all")
    with plt.style.context('ieee'):
        fig = plt.figure()
        
        plt.loglog(sizes**2, errsknown_median[3]*(sizes**2/sizes[3]**2)**(-1/2), 'k--', label='_nolegend_', lw=0.5)
        plt.loglog(sizes**2, errsknown_median, '.-b', label=r'known $\xi$ and $\zeta$')
    
        plt.loglog(sizes**2, errsAlgorithm1_median[3]*(sizes**2/sizes[3]**2)**(-1/2), 'k--', label='_nolegend_', lw=0.5)
        plt.loglog(sizes**2, errsAlgorithm1_median, '.--r', label='Algorithm 1')
        
        # plt.loglog(sizes**2, errs_no_median, ':g', label=r'no $\xi$ and $\zeta$')
    
        plt.legend()#loc=(0.5, 0.55))#, fontsize=6)
        
        plt.xlabel('Measurement size [pixels]')
        
        plt.ylabel('Median estimation error')
        fig.tight_layout()
        plt.show()

        plt.savefig(r'C:\Users\kreym\Google Drive\Thesis\Documents\Article\figures\error_convergence_experiment.pdf')
        