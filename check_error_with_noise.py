# -*- coding: utf-8 -*-
"""
Created on Sat Aug 29 20:04:26 2020

@author: kreym
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import scipy
import time

from fb_funcs import expand_fb, rot_img
from generate_clean_micrograph_2d import generate_clean_micrograph_2d_one_neighbor_rots
from funcs_calc_moments import M2_2d, M3_2d
from funcs_calc_moments_rot import calck1k2k3, calck1, calcS3_x, calcS3_x_neigh, calcS3_full_shift, calcS3_x_grad, calcS3_x_neigh_grad, calcS2_grad_full_shift, calcS2_x_grad, calcS2_x_neigh_grad, calcN_mat
from psf_functions_2d import full_psf_2d
from costgrad import check_moments
from optimization_funcs_rot import optimize_2d_known_psf_rot, optimize_2d_known_psf_mat_rot
from calc_estimation_error import calc_estimation_error
from generate_signal import generate_signal
from makeExtraMat import makeExtraMat
plt.close("all")

NumIters = 100
NumSigmas = 25
sigmas = np.logspace(np.log10(0.10/25), np.log10(40/25), num=NumSigmas)
est_errors = np.zeros((NumIters, NumSigmas))

L = 5

ne = 100

gamma_initial = 0.08

X_initial = np.random.rand(L, L)
X_initial = L*L * X_initial / np.linalg.norm(X_initial)
_, z_initial, _, _, _ = expand_fb(X_initial, ne)
        


for ii in range(NumIters):
    print(f'iteration #{ii}')
    X = np.random.rand(L, L)
    X = X / np.linalg.norm(X)

    
    N = 2000

    B, z, roots, kvals, nu = expand_fb(X, ne)
    Xrec = np.reshape(np.real(B @ z), np.shape(X))
    theta = np.random.uniform(0, 2*np.pi)
    Xrot = rot_img(theta, z, kvals, B)
    y, s, locs = generate_clean_micrograph_2d_one_neighbor_rots(z, kvals, B, L, N, 0.10*(N/L)**2)
    gamma = s[0]*(L/N)**2
    for sig in range(NumSigmas):
        print(f'sigma2 = {sigmas[sig]}')

        sigma2 = sigmas[sig]
        y_N = y + np.random.normal(loc=0, scale=np.sqrt(sigma2), size=np.shape(y))
        
        Bk = np.zeros((2*L-1, 2*L-1, nu), dtype=np.complex_)
        for i in range(nu):
            Bk[ :, :, i] = np.fft.fft2(np.pad(np.reshape(B[ :, i], (L, L)), L//2))
            
        yy = np.zeros((N, N, 1))
        yy[ :, :, 0] = y_N
        M1_y = np.mean(y_N)
        
        M2_y = np.zeros((L, L))
        for i1 in range(L):
            for j1 in range(L):
                M2_y[i1, j1] = M2_2d(yy, (j1, i1))
        
        M3_y = np.zeros((L, L, L, L))
        for i1 in range(L):
            for j1 in range(L):
                for i2 in range(L):
                    for j2 in range(L):
                        M3_y[i1, j1, i2, j2] = M3_2d(yy, (j1, i1), (j2, i2))
    
        psf = full_psf_2d(locs, L)

        X_est = optimize_2d_known_psf_mat_rot(np.concatenate((np.reshape(gamma_initial, (1,)), np.real(z), np.imag(z))), Bk, kvals, M1_y, M2_y, M3_y, sigma2, psf, L, 1, gtol=1e-6)
        
        gamma_est = X_est.x[0]
        z_est = X_est.x[1:nu+1] + 1j*X_est.x[nu+1:]
        X_est = np.reshape(np.real(B @ z_est), np.shape(X))
        est_errors[ii, sig] = calc_estimation_error(Xrec, X_est)

mean_errors = np.mean(est_errors, axis=0)
plt.figure()
plt.loglog(sigmas, mean_errors)
