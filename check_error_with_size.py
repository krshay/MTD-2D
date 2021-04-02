# -*- coding: utf-8 -*-
"""
Created on Fri Aug 21 14:34:51 2020

@author: kreym
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import scipy

from fb_funcs import expand_fb, rot_img
from generate_clean_micrograph_2d import generate_clean_micrograph_2d_one_neighbor_rots
from funcs_calc_moments import M3_2d
from funcs_calc_moments_rot import calck1k2k3, calcS3_x, calcS3_x_neigh, calcS3_full_shift, calcS3_x_grad, calcS3_x_neigh_grad
from psf_functions_2d import full_psf_2d
from costgrad import check_moments
from optimization_funcs_rot import optimize_2d_known_psf_rot
from calc_estimation_error import calc_estimation_error
from generate_signal import generate_signal
plt.close("all")



L = 5

N = 4000
ne = 5

k1k2k3_map = calck1k2k3(L)


NumOfSizes = 20
NumOfIters = 1
errs = np.zeros((NumOfIters, NumOfSizes))

sizes = np.linspace(0, N, NumOfSizes, dtype=int)
for ii in range(NumOfIters):
    print(ii)
    X = np.random.rand(5, 5)
    ne = 5
    B, z, roots, kvals, nu = expand_fb(X, ne)
    Xrec = np.reshape(np.real(B @ z), np.shape(X))
    theta = np.random.uniform(0, 2*np.pi)
    Xrot = rot_img(theta, z, kvals, B)
    y, s, locs = generate_clean_micrograph_2d_one_neighbor_rots(z, kvals, B, L, N, 0.10*(N/L)**2)
    gamma = s[0]*(L/N)**2
    sigma = 0
    y = y + np.random.normal(loc=0, scale=sigma, size=np.shape(y))
    
    Bk = np.zeros((2*L-1, 2*L-1, ne), dtype=np.complex_)
    for i in range(ne):
        Bk[ :, :, i] = np.fft.fft2(np.pad(np.reshape(B[ :, i], (L, L)), L//2))
        
    kmax = np.max(kvals)
    Nmax = 6*kmax
    S3_x, gS3_x = calcS3_x_grad(L, Nmax, Bk, z, kvals, k1k2k3_map)
    
    z0 = z[kvals==0]
    Nmax_neigh = 4*kmax
    S3_x_neigh, gS3_x_neigh = calcS3_x_neigh_grad(L, Nmax_neigh, Bk, z, kvals, k1k2k3_map)
    
    psf = full_psf_2d(locs, L)
    
    M3_y_check = np.zeros((L, L, L, L), dtype=np.complex_)
    for i1 in range(L):
        for j1 in range(L):
            for i2 in range(L):
                for j2 in range(L):
                    M3_y_check[i1, j1, i2, j2], _, _ = calcS3_full_shift((j1, i1), (j2, i2), S3_x, S3_x_neigh, L, psf) 
                    
    M3_y_check = gamma*np.real(M3_y_check)
    
    for sz in range(NumOfSizes):
        size = sizes[sz]
        print(size)
        yy = np.zeros((size, size, 1))
        yy[ :, :, 0] = y[:size, :size]
        M3_y = np.zeros((L, L, L, L))
        for i1 in range(L):
            for j1 in range(L):
                for i2 in range(L):
                    for j2 in range(L):
                        M3_y[i1, j1, i2, j2] = M3_2d(yy, (j1, i1), (j2, i2))
        
        errs[ii, sz] = np.sum((M3_y - M3_y_check)**2)
        
plt.figure()
plt.loglog(sizes, np.mean(errs, axis=0))