# -*- coding: utf-8 -*-
"""
Created on Sat Mar  6 20:15:39 2021

@author: kreym
"""
import numpy as np

import multiprocessing as mp

from funcs_calc_moments import M2_2d

from calc_3rdorder_ac import calc_3rdorder_ac

from generate_clean_micrograph_2d import generate_clean_micrograph_2d_rots

def calcM3_parallel(L, sigma2, gamma, c, kvals, Bk, W, T, N, NumMicrographs):
    ys = []
    M1_ys = np.zeros((NumMicrographs, ))
    M2_ys = np.zeros((NumMicrographs, L, L))
    M3_ys = np.zeros((NumMicrographs, L, L, L, L))
    for idx in range(NumMicrographs):
        y_clean, _, _ = generate_clean_micrograph_2d_rots(c, kvals, Bk, W, L, N, gamma*(N/L)**2, T)
        y = y_clean + np.random.default_rng().normal(loc=0, scale=np.sqrt(sigma2), size=np.shape(y_clean))
        
        yy = np.zeros((N, N, 1))
        yy[ :, :, 0] = y
        ys.append(yy)
        
        M1_y = np.mean(y)
        M1_ys[idx] = M1_y
        
        M2_y = np.zeros((L, L))
        for i1 in range(L):
            for j1 in range(L):
                M2_y[i1, j1] = M2_2d(yy, (i1, j1))
        M2_ys[idx, :, :] = M2_y
    
    num_cpus = mp.cpu_count()
    pool = mp.Pool(num_cpus)
    M3_ys_parallel = pool.starmap(calc_3rdorder_ac, [[L, y] for y in ys])
    pool.close()
    pool.join()
    
    for ii in range(NumMicrographs):
        M3_ys[ii, :, :, :, :] = M3_ys_parallel[ii]
    
    return M1_ys, M2_ys, M3_ys
