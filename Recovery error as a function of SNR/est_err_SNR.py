# -*- coding: utf-8 -*-
"""
Created on Thu Apr  1 12:02:30 2021

@author: kreym
"""

import numpy as np
import matplotlib.pyplot as plt

import multiprocessing as mp

from Utils.calc_err_SNR import calc_err_SNR

plt.close("all")

if __name__ == '__main__':
    N = 20000
    Niters = 30
    SNRs_length = 30
    L = 5
    ne = 10
    SNRs = np.logspace(np.log10(0.09), np.log10(100), SNRs_length)
    
    num_cpus = mp.cpu_count()
    pool = mp.Pool(num_cpus)
    S = pool.starmap(calc_err_SNR, [[L, ne, N, SNRs, i] for i in range(Niters)])
    pool.close()
    pool.join() 
    
    errs = np.zeros((Niters, SNRs_length))

    for j in range(Niters):
        errs[j, :] = S[j][0][np.arange(SNRs_length), np.argmin(S[j][1], axis=1)]
    errs_mean = np.median(errs, 0)
    with plt.style.context('ieee'):
        fig = plt.figure()
        plt.loglog(SNRs[4:18], errs_mean[4]*(SNRs[4:18]/SNRs[4])**(-3/4), 'k--', lw=0.5)
        plt.loglog(SNRs, errs_mean, '.-b')  
        plt.xlabel('SNR')
        plt.ylabel('Median estimation error')
        fig.tight_layout()
        plt.show()
        plt.savefig(r'C:\Users\kreym\Google Drive\Thesis\Documents\Article\figures\error_SNR_experiment.pdf')
