# -*- coding: utf-8 -*-
"""
Created on Thu Apr  1 12:02:30 2021

@author: kreym
"""

import numpy as np
import matplotlib.pyplot as plt
import time


import multiprocessing as mp

from calc_err_SNR import calc_err_SNR


plt.close("all")

if __name__ == '__main__':
    start = time.time()
    N = 30000
    Niters = 50
    SNRs_length = 20
    L = 5
    ne = 10
    SNRs = np.logspace(np.log10(0.1), np.log10(10), SNRs_length)
    
    num_cpus = mp.cpu_count()
    pool = mp.Pool(num_cpus)
    S = pool.starmap(calc_err_SNR, [[L, ne, N, SNRs, i] for i in range(Niters)])
    pool.close()
    pool.join() 
    
    finish = time.time() - start
    
    errs = np.zeros((Niters, 10))

    for j in range(Niters):
        errs[j, :] = np.min(S[j][0], axis=1)
    errs_mean = np.median(errs, 0)
    
    np.save('SNRs.npy', SNRs)
    
    np.save('errs_SNR_0_50.npy', errs)


    plt.figure()
    plt.loglog(SNRs, errs_mean)  
    # plt.loglog(SNrs, errs_mean[-1]*(sizes**2/sizes[-1]**2)**(-1/2), '--')
    plt.xlabel('SNR')
    plt.ylabel('MSE')
    plt.title('MSE in estimation of FB coefficients vs. micrograph size')
    # plt.legend(('data', '-1/2 slope'))
    