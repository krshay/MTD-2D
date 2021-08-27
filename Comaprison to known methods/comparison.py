# -*- coding: utf-8 -*-
"""
Created on Tue Aug 24 17:46:04 2021

@author: Shay Kreymer
"""

import numpy as np
import matplotlib.pyplot as plt
import shelve

import multiprocessing as mp

from Utils.calc_err_SNR import calc_err_SNR_comparison

plt.close("all")

if __name__ == '__main__':
    # Code to reproduce Fig. 6 in the paper. 
    # Estimation error as a function of SNR for both cases: known pair separation functions, and Algorithm 1.
    # %% Preliminary definitions
    N = 20000
    Niters = 10
    SNRs_length = 10
    L = 5
    ne = 10
    SNRs = np.logspace(np.log10(10**(-4)), np.log10(100), 10)
    
    # %% Calculations
    num_cpus = mp.cpu_count()
    
    pool = mp.Pool(num_cpus)
    S = pool.starmap(calc_err_SNR_comparison, [[L, ne, N, SNRs, i] for i in range(Niters)])
    pool.close()
    pool.join()
    
    errs_Algorithm1 = np.zeros((Niters, SNRs_length))

    for j in range(Niters):
        errs_Algorithm1[j, :] = S[j][0][np.arange(SNRs_length), np.argmin(S[j][1], axis=1)]
    errs_Algorithm1_mean = np.mean(errs_Algorithm1, 0)
    
    errs_conv = np.zeros((Niters, SNRs_length))

    for j in range(Niters):
        errs_conv[j, :] = S[j][2]
    errs_conv_mean = np.mean(errs_conv, 0)
    
    
    filename=r'C:/Users/kreym/Documents/GitHub/MTD-2D/Comaprison to known methods/shelve_25082021.out'
    # %% load
    my_shelf = shelve.open(filename)
    for key in my_shelf:
        globals()[key]=my_shelf[key]
    my_shelf.close()
    # %% Plotting
    plt.close("all")
    with plt.style.context('ieee'):
        fig = plt.figure()
        plt.loglog(SNRs, errs_Algorithm1_mean, '.-b', label='Algorithm 1')  
        plt.loglog(SNRs, errs_conv_mean, '.--r', label='oracle-deconv')
        plt.legend(loc=1)
        plt.xlabel('SNR')
        plt.ylabel('Mean estimation error')
        fig.tight_layout()
        plt.show()
        
        plt.savefig(r'C:\Users\kreym\Google Drive\Thesis\Documents\Article v2\figures\comparison_oracle.pdf')


