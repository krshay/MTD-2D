# -*- coding: utf-8 -*-
"""
Created on Sat Mar  6 20:17:45 2021

@author: kreym
"""
import numpy as np

from funcs_calc_moments import M3_2d

def calc_3rdorder_ac(L, yy):
    M3_y = np.zeros((L, L, L, L))
    for i1 in range(L):
        for j1 in range(L):
            for i2 in range(L):
                for j2 in range(L):
                    M3_y[i1, j1, i2, j2] = M3_2d(yy, (i1, j1), (i2, j2))
    return M3_y