#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 12 10:51:02 2021

@author: shaykreymer
"""

import numpy as np
import matplotlib.pyplot as plt
import time


import multiprocessing as mp

from calc_err_size import calc_err_size_nopsftsf


plt.close("all")

if __name__ == '__main__':
    start = time.time()
    N = 30000
    Niters = 100
    L = 5
    ne = 10
    sizes = np.logspace(np.log10(3000), np.log10(N), 10).astype(np.int)
    
    num_cpus = mp.cpu_count()
    pool = mp.Pool(num_cpus)
    S = pool.starmap(calc_err_size_nopsftsf, [[L, ne, N, sizes, i] for i in range(Niters)])
    pool.close()
    pool.join() 
    
    finish = time.time() - start
    
    errs = np.zeros((Niters, 10))

    for j in range(Niters):
        errs[j, :] = np.min(S[j][0], axis=1)
    errs_mean = np.median(errs, 0)
    
    np.save('sizes.npy', sizes)
    
    np.save('errs_no_50_100.npy', errs)
    
    # np.save('S_no_0_50.npy', np.array(S))


    plt.figure()
    plt.loglog(sizes**2, errs_mean)  
    plt.loglog(sizes**2, errs_mean[-1]*(sizes**2/sizes[-1]**2)**(-1/2), '--')
    plt.xlabel('# of pixels')
    plt.ylabel('MSE')
    plt.title('MSE in estimation of FB coefficients vs. micrograph size')
    plt.legend(('data', '-1/2 slope'))
    