# -*- coding: utf-8 -*-
"""
Created on Sat Aug  8 11:36:31 2020

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
plt.close("all")



L = 5
F = scipy.linalg.dft(L)
N = 1000
ne = 5
X = np.random.multivariate_normal(mean=np.zeros((L,)), cov=np.real(np.conjugate(np.transpose(F))*np.diag(np.array([5, 4, 3, 2, 1]))*F), size=(L, L))[:,:,0]
X = X - np.mean(X)
X = L*L * X / np.linalg.norm(X)

B, z, roots, kvals, nu = expand_fb(X, ne)
Bk = np.zeros((2*L-1, 2*L-1, ne), dtype=np.complex_)
for i in range(ne):
    Bk[ :, :, i] = np.fft.fft2(np.pad(np.reshape(B[ :, i], (L, L)), L//2))
    
k1k2k3_map = calck1k2k3(L)
kmax = np.max(kvals)


NumIters = 10000
M3_y = np.zeros((L, L, L, L, NumIters))
M3_y_check = np.zeros((L, L, L, L, NumIters))
M3_y_clean = np.zeros((L, L, L, L, NumIters))
M3_y_extra = np.zeros((L, L, L, L, NumIters))
gammas = np.zeros((NumIters, ))
y = np.zeros((N, N, NumIters))
psfs = np.zeros((4*L-3, 4*L-3, NumIters))
zs = np.zeros((ne, NumIters), dtype=np.complex_)

for n in range(NumIters):
    print(n)
    X = np.random.multivariate_normal(mean=np.zeros((L,)), cov=np.real(np.conjugate(np.transpose(F))*np.diag(np.array([5, 4, 3, 2, 1]))*F), size=(L, L))[:,:,0]
    X = X - np.mean(X)
    X = L*L * X / np.linalg.norm(X)
    
    _, z, _, _, _ = expand_fb(X, ne)
    zs[ :, n] = z
    y[ :, :, n], s, locs = generate_clean_micrograph_2d_one_neighbor_rots(z, kvals, B, L, N, 0.05*(N/L)**2)
    gammas[n] = s[0]*(L/N)**2
    
    yy = np.zeros((N, N, 1))
    yy[ :, :, 0] = y[ :, :, n]
    for i1 in range(L):
        for j1 in range(L):
            for i2 in range(L):
                for j2 in range(L):
                    M3_y[i1, j1, i2, j2, n] = M3_2d(yy, (j1, i1), (j2, i2))
    
    Nmax = 6*kmax
    S3_x = calcS3_x(L, Nmax, Bk, z, kvals, k1k2k3_map)
    Nmax_neigh = 4*kmax
    S3_x_neigh = calcS3_x_neigh(L, Nmax_neigh, Bk, z, kvals, k1k2k3_map)
    
    psfs[ :, :, n] = full_psf_2d(locs, L)
    
    for i1 in range(L):
        for j1 in range(L):
            for i2 in range(L):
                for j2 in range(L):
                    M3_y_check[i1, j1, i2, j2, n], M3_y_clean[i1, j1, i2, j2, n], M3_y_extra[i1, j1, i2, j2, n] = calcS3_full_shift((j1, i1), (j2, i2), S3_x, S3_x_neigh, L, psfs[ :, :, n]) 
                    
    M3_y_check[ :, :, :, :, n] = gammas[n]*M3_y_check[ :, :, :, :, n]
