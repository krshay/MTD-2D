# -*- coding: utf-8 -*-
"""
Created on Fri Aug 28 20:58:08 2020

@author: kreym
"""
import numpy as np
import scipy

def makeExtraMat(L, psf):
    # Rearranging the pair separation function to a matrix-form, to ease calculations
    Mat3 = scipy.sparse.lil_matrix((L**4, (2*L-1)**4))
    for i1 in range(L):
        for j1 in range(L):
            for i2 in range(L):
                for j2 in range(L):
                    shift1y = j1
                    shift1x = i1
                    shift2y = j2
                    shift2x = i2
                            
                    row = np.ravel_multi_index([i1, j1, i2, j2], (L, L, L, L))
                    
                    for j in range(max(shift1y, shift2y) - (L-1), L + min(shift1y, shift2y)):
                        for i in range(max(shift1x, shift2x) - (L-1), L + min(shift1x, shift2x)):
                            if not (np.abs(i) < L and np.abs(j) < L):
                                Mat3[row, np.ravel_multi_index([(shift2x-shift1x)%(2*L-1), (shift2y-shift1y)%(2*L-1), (i-shift1x)%(2*L-1), (j-shift1y)%(2*L-1)], (2*L-1, 2*L-1, 2*L-1, 2*L-1))] += psf[i + 2*L-2, j + 2*L-2]
                    
                    for j in range(shift1y - (L-1), L + shift1y - shift2y):
                        for i in range(shift1x - (L-1), L + shift1x - shift2x):
                            if not (np.abs(i) < L and np.abs(j) < L):
                                Mat3[row, np.ravel_multi_index([shift2x%(2*L-1), shift2y%(2*L-1), (shift1x-i)%(2*L-1), (shift1y-j)%(2*L-1)], (2*L-1, 2*L-1, 2*L-1, 2*L-1))] += psf[i + 2*L-2, j + 2*L-2]
                    
                    for j in range(shift2y - (L-1), L + shift2y - shift1y):
                        for i in range(shift2x - (L-1), L + shift2x - shift1x):
                            if not (np.abs(i) < L and np.abs(j) < L):
                                Mat3[row, np.ravel_multi_index([shift1x%(2*L-1), shift1y%(2*L-1), (shift2x-i)%(2*L-1), (shift2y-j)%(2*L-1)], (2*L-1, 2*L-1, 2*L-1, 2*L-1))] += psf[i + 2*L-2, j + 2*L-2]
    
    Mat2 = scipy.sparse.lil_matrix((L**2, (2*L-1)**2))
    for i1 in range(L):
        for j1 in range(L):
            shift1y = j1
            shift1x = i1
            
            row = np.ravel_multi_index([i1, j1], (L, L))
            
            for j in range(shift1y-(L-1), L + shift1y):
                for i in range(shift1x-(L-1), L + shift1x):
                    if not (np.abs(i) < L and np.abs(j) < L):
                        Mat2[row, np.ravel_multi_index([(i-shift1x)%(2*L-1), (j-shift1y)%(2*L-1)], (2*L-1, 2*L-1))] += psf[i + 2*L-2, j + 2*L-2]
                        
    return scipy.sparse.csr_matrix(Mat2), scipy.sparse.csr_matrix(Mat3)