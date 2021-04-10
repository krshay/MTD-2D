# -*- coding: utf-8 -*-
"""
Created on Sat Mar  6 20:15:39 2021

@author: Shay Kreymer
"""
import numpy as np

import multiprocessing as mp

from Utils.funcs_calc_moments import M2_2d

from Utils.calc_3rdorder_ac import calc_M3_for_micrograph, calc_M3_for_list

from Utils.generate_clean_micrograph_2d import generate_clean_micrograph_2d_rots

import itertools

def calcM3_parallel_micrographs(L, sigma2, gamma, c, kvals, Bk, W, T, N, NumMicrographs):
    print('Started calculations')
    M1_ys = np.zeros((NumMicrographs, ))
    M2_ys = np.zeros((NumMicrographs, L, L))
    M3_ys = np.zeros((NumMicrographs, L, L, L, L))

    num_cpus = mp.cpu_count()
    pool = mp.Pool(num_cpus)
    Ms_parallel = pool.starmap(calc_M3_for_micrograph, [[L, c, kvals, Bk, W, N, gamma, T, sigma2, ii] for ii in range(NumMicrographs)])
    pool.close()
    pool.join()
    
    for ii in range(NumMicrographs):
        M1_ys[ii] = Ms_parallel[ii][0]
        M2_ys[ii, :, :] = Ms_parallel[ii][1]
        M3_ys[ii, :, :, :, :] = Ms_parallel[ii][2]
    
    return M1_ys, M2_ys, M3_ys

def calcM3_parallel_shifts(L, sigma2, gamma, c, kvals, Bk, W, T, N):
    print('Started calculations')
    y_clean, _, _ = generate_clean_micrograph_2d_rots(c, kvals, Bk, W, L, N, gamma*(N/L)**2, T, seed=100)
    y = y_clean + np.random.default_rng().normal(loc=0, scale=np.sqrt(sigma2), size=np.shape(y_clean))
    yy = np.zeros((N, N, 1))
    yy[ :, :, 0] = y
    M1_y = np.mean(y)
    
    M2_y = np.zeros((L, L))
    for i1 in range(L):
        for j1 in range(L):
            M2_y[i1, j1] = M2_2d(yy, (i1, j1))
    print('finished autocorrelations 1st and 2nd order')
    
    num_cpus = mp.cpu_count()
    list_shifts = list(itertools.product(np.arange(L), np.arange(L), np.arange(L), np.arange(L)))
    list_list_shifts = np.array_split(list_shifts, num_cpus)
    pool = mp.Pool(num_cpus)
    M3_ys_parallel = pool.starmap(calc_M3_for_list, [[yy, list_list_shifts[ii], ii] for ii in range(num_cpus)])
    pool.close()
    pool.join()
    
    M3_y = []
    for M3_ys in M3_ys_parallel:
        values_M3_ys = list(M3_ys.values())
        M3_y = M3_y + values_M3_ys
    M3_y = np.reshape(np.array(M3_y), (L, L, L, L))

    return M1_y, M2_y, M3_y

