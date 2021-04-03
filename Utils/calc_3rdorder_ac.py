# -*- coding: utf-8 -*-
"""
Created on Sat Mar  6 20:17:45 2021

@author: kreym
"""
import numpy as np

import multiprocessing as mp

from Utils.funcs_calc_moments import M3_2d

def calc_M3_for_micrograph(L, yy):
    # Calculations of third-order autocorrelation of micrograph yy, for all possible shifts up to L - 1
    M3_y = np.zeros((L, L, L, L))
    for i1 in range(L):
        print(i1)
        for j1 in range(L):
            for i2 in range(L):
                for j2 in range(L):
                    M3_y[i1, j1, i2, j2] = M3_2d(yy, (i1, j1), (i2, j2))
    return M3_y

def calc_M3_for_list(yy, list_shifts, ii):
    # Calculations of third-order autocorrelation of micrograph yy, for shifts in list_shifts
    M3_y = {}
    for shift_idx in range(np.shape(list_shifts)[0]):
        i1, j1, i2, j2 = list_shifts[shift_idx]
        M3_y[(i1, j1, i2, j2)] = M3_2d(yy, (i1, j1), (i2, j2))
    print(f'finished part #{ii} out of {mp.cpu_count()}')
    return M3_y