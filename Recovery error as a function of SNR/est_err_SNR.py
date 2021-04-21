# -*- coding: utf-8 -*-
"""
Created on Thu Apr  1 12:02:30 2021

@author: kreym
"""

import numpy as np
import matplotlib.pyplot as plt

import multiprocessing as mp

from Utils.calc_err_SNR import calc_err_SNR_Algorithm1, calc_err_SNR_knownpsftsf


plt.close("all")

if __name__ == '__main__':
    N = 10000
    Niters = 50
    SNRs_length = 30
    L = 5
    ne = 10
    SNRs = np.logspace(np.log10(0.25), np.log10(1000), SNRs_length)
    
    num_cpus = mp.cpu_count()
    
    pool = mp.Pool(num_cpus)
    S_known = pool.starmap(calc_err_SNR_knownpsftsf, [[L, ne, N, SNRs, i] for i in range(Niters)])
    pool.close()
    pool.join()
    
    errs_known = np.zeros((Niters, SNRs_length))

    for j in range(Niters):
        errs_known[j, :] = S_known[j][0][np.arange(SNRs_length), np.argmin(S_known[j][1], axis=1)]
    errs_known_mean = np.median(errs_known, 0)
    
    pool = mp.Pool(num_cpus)
    S_Algorithm1 = pool.starmap(calc_err_SNR_Algorithm1, [[L, ne, N, SNRs, i] for i in range(Niters)])
    pool.close()
    pool.join()
    
    errs_Algorithm1 = np.zeros((Niters, SNRs_length))

    for j in range(Niters):
        errs_Algorithm1[j, :] = S_Algorithm1[j][0][np.arange(SNRs_length), np.argmin(S_Algorithm1[j][1], axis=1)]
    errs_Algorithm1_mean = np.median(errs_Algorithm1, 0)
    
    # with plt.style.context('ieee'):
    fig = plt.figure()
    plt.loglog(SNRs, errs_known_mean, '.-b')  
    plt.loglog(SNRs[0:14], errs_known_mean[3]*(SNRs[0:14]/SNRs[3])**(-3/2), 'k--', lw=0.5)
    
    plt.loglog(SNRs, errs_Algorithm1_mean, '.--r')  
    plt.loglog(SNRs[0:14], errs_Algorithm1_mean[3]*(SNRs[0:14]/SNRs[3])**(-3/4), 'k--', lw=0.5)
    
    plt.xlabel('SNR')
    plt.ylabel('Median estimation error')
    fig.tight_layout()
    plt.show()
        # plt.savefig(r'C:\Users\kreym\Google Drive\Thesis\Documents\Article\figures\error_SNR_experiment.pdf')
