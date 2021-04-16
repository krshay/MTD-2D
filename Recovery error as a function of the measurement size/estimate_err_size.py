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
    Niters = 20
    L = 5
    ne = 10
    Nsizes = 15
    sizes = np.logspace(np.log10(1000), np.log10(N), 15).astype(int)
    
    num_cpus = mp.cpu_count()
    # %% Known psf and tsf
    pool = mp.Pool(num_cpus)
    Sknown = pool.starmap(calc_err_size_knownpsftsf, [[L, ne, N, sizes, i] for i in range(Niters)])
    pool.close()
    pool.join() 
    
    np.save('Sknown.npy', np.array(Sknown))
    
    # %% Algorithm1
    pool = mp.Pool(num_cpus)
    SAlgorithm1 = pool.starmap(calc_err_size_Algorithm1, [[L, ne, N, sizes, i] for i in range(Niters)])
    pool.close()
    pool.join()
    
    np.save('SAlgorithm1.npy', np.array(SAlgorithm1))
    
    # %% No psf and tsf
    pool = mp.Pool(num_cpus)
    Sno = pool.starmap(calc_err_size_nopsftsf, [[L, ne, N, sizes, i] for i in range(Niters)])
    pool.close()
    pool.join()
    
    np.save('Sno.npy', np.array(Sno))


    