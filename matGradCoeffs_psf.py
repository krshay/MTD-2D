# -*- coding: utf-8 -*-
"""
Created on Sat Oct 31 16:45:16 2020

@author: kreym
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Aug 28 20:58:08 2020

@author: kreym
"""
import numpy as np
import scipy
import scipy.special as special

def matGradCoeffs_psf(L, N_coeffs, roots, b, S3_x_neigh):
    Mat = np.zeros((L**4, (2*L-1)**4, N_coeffs), dtype=np.complex_)
    for i1 in range(L):
        for j1 in range(L):
            for i2 in range(L):
                for j2 in range(L):
                    shift1y = j1
                    shift1x = i1
                    shift2y = j2
                    shift2x = i2
                            
                    row = np.ravel_multi_index([j1, i1, j2, i2], (L, L, L, L))
                    
                    for j in range(max(shift1y, shift2y) - (L-1), L + min(shift1y, shift2y)):
                        for i in range(max(shift1x, shift2x) - (L-1), L + min(shift1x, shift2x)):
                            if not (np.abs(i) < L and np.abs(j) < L):
                                Js = np.array([special.j0(roots[n]*np.sqrt(i**2 + j**2)/b) for n in range(N_coeffs)])
                                Mat[row, np.ravel_multi_index([(shift2y-shift1y)%(2*L-1), (shift2x-shift1x)%(2*L-1), (j-shift1y)%(2*L-1), (i-shift1x)%(2*L-1)], (2*L-1, 2*L-1, 2*L-1, 2*L-1)), :] = Mat[row, np.ravel_multi_index([(shift2y-shift1y)%(2*L-1), (shift2x-shift1x)%(2*L-1), (j-shift1y)%(2*L-1), (i-shift1x)%(2*L-1)], (2*L-1, 2*L-1, 2*L-1, 2*L-1)), :] + S3_x_neigh[shift2y-shift1y, shift2x-shift1x, j-shift1y, i-shift1x] * Js
                    
                    for j in range(shift1y - (L-1), L + shift1y - shift2y):
                        for i in range(shift1x - (L-1), L + shift1x - shift2x):
                            if not (np.abs(i) < L and np.abs(j) < L):
                                Js = np.array([special.j0(roots[n]*np.sqrt(i**2 + j**2)/b) for n in range(N_coeffs)])
                                Mat[row, np.ravel_multi_index([shift2y%(2*L-1), shift2x%(2*L-1), (shift1y-j)%(2*L-1), (shift1x-i)%(2*L-1)], (2*L-1, 2*L-1, 2*L-1, 2*L-1)), :] = Mat[row, np.ravel_multi_index([shift2y%(2*L-1), shift2x%(2*L-1), (shift1y-j)%(2*L-1), (shift1x-i)%(2*L-1)], (2*L-1, 2*L-1, 2*L-1, 2*L-1)), :] + S3_x_neigh[shift2y, shift2x, shift1y-j, shift1x-i] * Js
                    
                    for j in range(shift2y - (L-1), L + shift2y - shift1y):
                        for i in range(shift2x - (L-1), L + shift2x - shift1x):
                            if not (np.abs(i) < L and np.abs(j) < L):
                                Js = np.array([special.j0(roots[n]*np.sqrt(i**2 + j**2)/b) for n in range(N_coeffs)])
                                Mat[row, np.ravel_multi_index([shift1y%(2*L-1), shift1x%(2*L-1), (shift2y-j)%(2*L-1), (shift2x-i)%(2*L-1)], (2*L-1, 2*L-1, 2*L-1, 2*L-1)), :] = Mat[row, np.ravel_multi_index([shift1y%(2*L-1), shift1x%(2*L-1), (shift2y-j)%(2*L-1), (shift2x-i)%(2*L-1)], (2*L-1, 2*L-1, 2*L-1, 2*L-1)), :] + S3_x_neigh[shift1y, shift1x, shift2y-j, shift2x-i] * Js

    return np.sum(Mat, 1)