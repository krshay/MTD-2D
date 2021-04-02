# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 14:27:49 2020

@author: kreym
"""

import numpy as np
import matplotlib.pyplot as plt
import time

from fb_funcs import expand_fb, min_err_rots, min_err_coeffs, calcT
from generate_clean_micrograph_2d import generate_clean_micrograph_2d_one_neighbor_rots
from funcs_calc_moments import M2_2d, M3_2d
from psf_functions_2d import full_psf_2d, coeffs_initial_guess
import optimization_funcs_rot
from makeExtraMat import makeExtraMat


plt.close("all")

if __name__ == '__main__':
    start = time.time()
    N = 15000
    Niters = 100
    sizes = np.logspace(np.log10(3000), np.log10(N), 20).astype(np.int)
    sizes = np.array([3500, 6000, 7500, 10000, 12000, 13500, 15000])#, 27000, 32000, 35000, 40000]
    errs = np.zeros((Niters, len(sizes)))
    
    for i in range(Niters):
        X = np.random.rand(5, 5)
        L = np.shape(X)[0]
        X = X / np.linalg.norm(X)
        
        W = 2*L-1 # L for arbitrary spacing distribution, 2*L-1 for well-separated
        
        ne = 9
        B, z, roots, kvals, nu = expand_fb(X, ne)
        T = calcT(nu, kvals)
        BT = B @ T.H
        c = np.real(T @ z)
        z = T.H@c
        Xrec = np.reshape(np.real(B @ z), np.shape(X))
        Bk = np.zeros((2*L-1, 2*L-1, nu), dtype=np.complex_)
        for ii in range(nu):
            Bk[ :, :, ii] = np.fft.fft2(np.pad(np.reshape(B[ :, ii], (L, L)), L//2))
    
        kmax = np.max(kvals)
        y_clean, s, locs = generate_clean_micrograph_2d_one_neighbor_rots(c, kvals, Bk, W, L, N, 0.045*(N/L)**2, T)
        gamma = s[0]*(L/N)**2
        sigma2 = 0
        y = y_clean + np.random.default_rng().normal(loc=0, scale=np.sqrt(sigma2), size=np.shape(y_clean))
    
        yy_all = np.zeros((N, N, 1))
        yy_all[ :, :, 0] = y
        
        
        psf = full_psf_2d(locs, L)
        ExtraMat = makeExtraMat(L, psf)
        
        gamma_initial = 0.040
        y_init, s_init, locs_init = generate_clean_micrograph_2d_one_neighbor_rots(c, kvals, Bk, W, L, N, gamma*(N/L)**2, T)
        psf_init = full_psf_2d(locs_init, L)
        psf_coeffs_initial = coeffs_initial_guess(L, locs_init, 20)
        X_initial = np.random.rand(L, L)
        X_initial = X_initial / np.linalg.norm(X_initial)
        
        _, z_initial, _, _, _ = expand_fb(X_initial, ne)
        c_initial = np.real(T @ z_initial)
        
    
        for (idx, sz) in enumerate(sizes):
            print(f'Micrograph sizes of {sz} * {sz}, iteration #{i+1}')
            yy = yy_all[ :sz, :sz, :]
            
            M1_y = np.mean(yy)
            
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
        
            X_est = optimization_funcs_rot.optimize_2d_known_psf_mat_rot_gamma(np.concatenate((np.reshape(gamma_initial, (1,)), c_initial)), Bk, T, kvals, M1_y, M2_y, M3_y, sigma2, psf, L, 1, ExtraMat, gtol=1e-15)
            gamma_est = X_est.x[0] 
            c_est = X_est.x[1:]
            z_est = T.H @ c_est
            est_err = min_err_rots(z, z_est, kvals, B, L)
            est_err_coeffs = min_err_coeffs(z, z_est, kvals)
            errs[i, idx] = est_err_coeffs[0]
    
    finish = time.time() - start
    
    errs_mean = np.mean(errs, axis=0)
    plt.figure()
    plt.loglog(sizes**2, errs_mean)  
    plt.loglog(sizes**2, errs_mean[-1]*(sizes**2/sizes[-1]**2)**(-1/2), '--')
    plt.xlabel('# of pixels')
    plt.ylabel('MSE')
    plt.title('MSE in estimation of FB coefficients vs. micrograph size')
    plt.legend(('data', '-1/2 slope'))
    