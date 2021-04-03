# -*- coding: utf-8 -*-
"""
Created on Fri Mar 12 20:22:57 2021

@author: kreym
"""

import numpy as np
import scipy
import itertools

import multiprocessing as mp

def maketsfMat_parallel(L, tsf):
    # Rearranging the triplet separation function to a matrix-form, to ease calculations. Parallel processing.
    shifts = list(itertools.product(np.arange(L), np.arange(L), np.arange(L), np.arange(L)))
    num_cpus = mp.cpu_count()
    divided_shifts = np.array_split(shifts, num_cpus)
    pool = mp.Pool(num_cpus)
    tsfMats = pool.starmap(maketsfMat_partial, [[L, tsf, shift_divided] for shift_divided in divided_shifts])
    pool.close()
    pool.join()
    
    return np.sum(tsfMats)

def maketsfMat_partial(L, tsf, shifts):
    Mat3 = scipy.sparse.lil_matrix((L**4, (2*L-1)**4))
    for shift in shifts:
        shift1y, shift1x, shift2y, shift2x = shift
                
        row = np.ravel_multi_index([shift1y, shift1x, shift2y, shift2x], (L, L, L, L))
        
        for j1 in range(shift1y - (L-1), L + shift1y):
            for i1 in range(shift1x - (L-1), L + shift1x):
                if not (np.abs(i1) < L and np.abs(j1) < L):
                    for j2 in range(max(j1, shift1y) - (L-1) - shift2y, L + min(j1, shift1y) - shift2y):
                        for i2 in range(max(i1, shift1x) - (L-1) - shift2x, L + min(i1, shift1x) - shift2x):
                            if not (np.abs(i2) < L and np.abs(j2) < L):
                                if not (np.abs(i2 - i1) < L and np.abs(j2 - j1) < L):
                                    Mat3[row, np.ravel_multi_index([(i1-shift1x)%(2*L-1), (j1-shift1y)%(2*L-1), (i2+shift2x-shift1x)%(2*L-1), (j2+shift2y-shift1y)%(2*L-1)], (2*L-1, 2*L-1, 2*L-1, 2*L-1))] += tsf[i1 + 2*L-2, j1 + 2*L-2, i2 + 2*L-2, j2 + 2*L-2]
        
        for j1 in range(shift2y - (L-1), L + shift2y):
            for i1 in range(shift2x - (L-1), L + shift2x):
                if not (np.abs(i1) < L and np.abs(j1) < L):
                    for j2 in range(max(j1, shift2y) - (L-1) - shift1y, L + min(j1, shift2y) - shift1y):
                        for i2 in range(max(i1, shift2x) - (L-1) - shift1x, L + min(i1, shift2x) - shift1x):
                            if not (np.abs(i2) < L and np.abs(j2) < L):
                                if not (np.abs(i2 - i1) < L and np.abs(j2 - j1) < L):
                                    Mat3[row, np.ravel_multi_index([(i1-shift2x)%(2*L-1), (j1-shift2y)%(2*L-1), (i2+shift1x-shift2x)%(2*L-1), (j2+shift1y-shift2y)%(2*L-1)], (2*L-1, 2*L-1, 2*L-1, 2*L-1))] += tsf[i1 + 2*L-2, j1 + 2*L-2, i2 + 2*L-2, j2 + 2*L-2]
        
    return scipy.sparse.csr_matrix(Mat3)