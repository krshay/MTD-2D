# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 14:27:49 2020

@author: kreym
"""

import numpy as np
import matplotlib.pyplot as plt
import time


import multiprocessing as mp

from calc_err_size import calc_err_size


plt.close("all")

if __name__ == '__main__':
    start = time.time()
    N = 30000
    Niters = 125
    L = 5
    ne = 10
    sizes = np.logspace(np.log10(3000), np.log10(N), 10).astype(np.int)
    
    num_cpus = mp.cpu_count()
    pool = mp.Pool(num_cpus)
    S = pool.starmap(calc_err_size, [[L, ne, N, sizes, i] for i in range(Niters)])
    pool.close()
    pool.join() 
    
    finish = time.time() - start
    
    errs = np.zeros((Niters, 10))

    for j in range(Niters):
        errs[j, :] = S[j]
    errs_mean = np.median(errs, 0)
    
    np.save('sizes.npy', sizes)
    
    np.save('errs3.npy', errs)


    plt.figure()
    plt.loglog(sizes**2, errs_mean)  
    plt.loglog(sizes**2, errs_mean[-1]*(sizes**2/sizes[-1]**2)**(-1/2), '--')
    plt.xlabel('# of pixels')
    plt.ylabel('MSE')
    plt.title('MSE in estimation of FB coefficients vs. micrograph size')
    plt.legend(('data', '-1/2 slope'))
    