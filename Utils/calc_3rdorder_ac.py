# -*- coding: utf-8 -*-
"""
Created on Sat Mar  6 20:17:45 2021

@author: Shay Kreymer
"""
import numpy as np

import multiprocessing as mp

from Utils.funcs_calc_moments import M2_2d, M3_2d

from Utils.generate_clean_micrograph_2d import generate_clean_micrograph_2d_rots

def calc_M3_for_micrograph(L, c, kvals, Bk, W, N, gamma, T, sigma2, idx):
    y_clean, _, _ = generate_clean_micrograph_2d_rots(c, kvals, Bk, W, L, N, gamma*(N/L)**2, T, seed=idx)
    y = y_clean + np.random.default_rng().normal(loc=0, scale=np.sqrt(sigma2), size=np.shape(y_clean))
    yy = np.zeros((N, N, 1))
    yy[ :, :, 0] = y
    
    M1_y = np.mean(y)
    
    M2_y = np.zeros((L, L))
    for i1 in range(L):
        for j1 in range(L):
            M2_y[i1, j1] = M2_2d(yy, (i1, j1))
            
    M3_y = np.zeros((L, L, L, L))
    for i1 in range(L):
        for j1 in range(L):
            for i2 in range(L):
                for j2 in range(L):
                    M3_y[i1, j1, i2, j2] = M3_2d(yy, (i1, j1), (i2, j2))
    print(f'finished micrograph #{idx}')
    return M1_y, M2_y, M3_y

def calc_M3_for_list(yy, list_shifts, ii):
    M3_y = {}
    for shift_idx in range(np.shape(list_shifts)[0]):
        i1, j1, i2, j2 = list_shifts[shift_idx]
        M3_y[(i1, j1, i2, j2)] = M3_2d(yy, (i1, j1), (i2, j2))
    print(f'finished part #{ii} out of {mp.cpu_count()}')
    return M3_y