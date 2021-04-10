# -*- coding: utf-8 -*-
"""
Created on Sun Jan 24 18:38:45 2021

@author: Shay Kreymer
"""

import numpy as np

from Utils.fb_funcs import expand_fb, min_err_coeffs, calcT
from Utils.generate_clean_micrograph_2d import generate_clean_micrograph_2d_rots
from Utils.funcs_calc_moments import M2_2d, M3_2d
from Utils.psf_tsf_funcs import full_psf_2d
from Utils.psf_tsf_funcs import full_tsf_2d
import Utils.optimization_funcs_rot
from Utils.psf_tsf_funcs import makeExtraMat
from Utils.psf_tsf_funcs import maketsfMat

def calc_err_size_knownpsftsf(L, ne, N, sizes, sd):
    # Calculation of estimation error in estimating a specific target image, multiple micrograph sizes. For the case of known PSF and TSF.
    np.random.seed(sd)
    errs = np.zeros((len(sizes), 3))
    costs = np.zeros((len(sizes), 3))
    X = np.random.rand(L, L)
    X = X / np.linalg.norm(X)
    
    W = L # L for arbitrary spacing distribution, 2*L-1 for well-separated

    B, z, roots, kvals, nu = expand_fb(X, ne)
    T = calcT(nu, kvals)
    c = np.real(T @ z)
    z = T.H@c
    Bk = np.zeros((2*L-1, 2*L-1, nu), dtype=np.complex_)
    for ii in range(nu):
        Bk[ :, :, ii] = np.fft.fft2(np.pad(np.reshape(B[ :, ii], (L, L)), L//2))


    gamma_initial = 0.09

    X_initial1 = np.random.rand(L, L)
    X_initial1 = X_initial1 / np.linalg.norm(X_initial1)
    
    _, z_initial1, _, _, _ = expand_fb(X_initial1, ne)
    c_initial1 = np.real(T @ z_initial1)

    X_initial2 = np.random.rand(L, L)
    X_initial2 = X_initial2 / np.linalg.norm(X_initial2)
    
    _, z_initial2, _, _, _ = expand_fb(X_initial2, ne)
    c_initial2 = np.real(T @ z_initial2)
    
    X_initial3 = np.random.rand(L, L)
    X_initial3 = X_initial3 / np.linalg.norm(X_initial3)
    
    _, z_initial3, _, _, _ = expand_fb(X_initial3, ne)
    c_initial3 = np.real(T @ z_initial3)

    for (idx, sz) in enumerate(sizes):
        y_clean, s, locs = generate_clean_micrograph_2d_rots(c, kvals, Bk, W, L, sz, 0.100*(sz/L)**2, T)
        sigma2 = 0
        y = y_clean + np.random.default_rng().normal(loc=0, scale=np.sqrt(sigma2), size=np.shape(y_clean))
        
        psf = full_psf_2d(locs, L)
        tsf = full_tsf_2d(locs, L)
        
        ExtraMat2, ExtraMat3 = makeExtraMat(L, psf)
        tsfMat = maketsfMat(L, tsf)
        yy = np.zeros((sz, sz, 1))
        yy[ :, :, 0] = y
        
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
    
        X_est1 = Utils.optimization_funcs_rot.optimize_2d_known_psf_triplets(np.concatenate((np.reshape(gamma_initial, (1,)), c_initial1)), Bk, T, kvals, M1_y, M2_y, M3_y, sigma2, psf, L, 1, tsfMat, ExtraMat2, ExtraMat3) 
        c_est1 = X_est1.x[1:]
        z_est1 = T.H @ c_est1
        est_err_coeffs1 = min_err_coeffs(z, z_est1, kvals)
        errs[idx, 0] = est_err_coeffs1[0]
        costs[idx, 0] = X_est1.fun
        
        X_est2 = Utils.optimization_funcs_rot.optimize_2d_known_psf_triplets(np.concatenate((np.reshape(gamma_initial, (1,)), c_initial2)), Bk, T, kvals, M1_y, M2_y, M3_y, sigma2, psf, L, 1, tsfMat, ExtraMat2, ExtraMat3) 
        c_est2 = X_est2.x[1:]
        z_est2 = T.H @ c_est2
        est_err_coeffs2 = min_err_coeffs(z, z_est2, kvals)
        errs[idx, 1] = est_err_coeffs2[0]
        costs[idx, 1] = X_est2.fun
        
        X_est3 = Utils.optimization_funcs_rot.optimize_2d_known_psf_triplets(np.concatenate((np.reshape(gamma_initial, (1,)), c_initial3)), Bk, T, kvals, M1_y, M2_y, M3_y, sigma2, psf, L, 1, tsfMat, ExtraMat2, ExtraMat3) 
        c_est3 = X_est3.x[1:]
        z_est3 = T.H @ c_est3
        est_err_coeffs3 = min_err_coeffs(z, z_est3, kvals)
        errs[idx, 2] = est_err_coeffs3[0]
        costs[idx, 2] = X_est3.fun
    return errs, costs

def calc_err_size_Algorithm1(L, ne, N, sizes, sd):
    # Calculation of estimation error in estimating a specific target image, multiple micrograph sizes. For the case of Algorithm 1.
    np.random.seed(sd)
    errs = np.zeros((len(sizes), 3))
    costs = np.zeros((len(sizes), 3))
    X = np.random.rand(L, L)
    X = X / np.linalg.norm(X)
    
    W = L # L for arbitrary spacing distribution, 2*L-1 for well-separated

    B, z, roots, kvals, nu = expand_fb(X, ne)
    T = calcT(nu, kvals)
    c = np.real(T @ z)
    z = T.H@c
    Bk = np.zeros((2*L-1, 2*L-1, nu), dtype=np.complex_)
    for ii in range(nu):
        Bk[ :, :, ii] = np.fft.fft2(np.pad(np.reshape(B[ :, ii], (L, L)), L//2))

    gamma_initial = 0.09

    X_initial1 = np.random.rand(L, L)
    X_initial1 = X_initial1 / np.linalg.norm(X_initial1)
    
    _, z_initial1, _, _, _ = expand_fb(X_initial1, ne)
    c_initial1 = np.real(T @ z_initial1)

    X_initial2 = np.random.rand(L, L)
    X_initial2 = X_initial2 / np.linalg.norm(X_initial2)
    
    _, z_initial2, _, _, _ = expand_fb(X_initial2, ne)
    c_initial2 = np.real(T @ z_initial2)
    
    X_initial3 = np.random.rand(L, L)
    X_initial3 = X_initial3 / np.linalg.norm(X_initial3)
    
    _, z_initial3, _, _, _ = expand_fb(X_initial3, ne)
    c_initial3 = np.real(T @ z_initial3)
    
    for (idx, sz) in enumerate(sizes):
        y_clean, s, locs = generate_clean_micrograph_2d_rots(c, kvals, Bk, W, L, sz, 0.100*(sz/L)**2, T)
        sigma2 = 0
        y = y_clean + np.random.default_rng().normal(loc=0, scale=np.sqrt(sigma2), size=np.shape(y_clean))
        
        yy = np.zeros((sz, sz, 1))
        yy[ :, :, 0] = y
        
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
    
        X_est1, _, _ = Utils.optimization_funcs_rot.optimize_rot_Algorithm1_notparallel(np.concatenate((np.reshape(gamma_initial, (1,)), c_initial1)), Bk, T, kvals, M1_y, M2_y, M3_y, sigma2, L, 1, W, 10000) 
        X_est2, _, _ = Utils.optimization_funcs_rot.optimize_rot_Algorithm1_notparallel(np.concatenate((np.reshape(gamma_initial, (1,)), c_initial2)), Bk, T, kvals, M1_y, M2_y, M3_y, sigma2, L, 1, W, 10000) 
        X_est3, _, _ = Utils.optimization_funcs_rot.optimize_rot_Algorithm1_notparallel(np.concatenate((np.reshape(gamma_initial, (1,)), c_initial3)), Bk, T, kvals, M1_y, M2_y, M3_y, sigma2, L, 1, W, 10000) 

        c_est1 = X_est1.x[1:]
        z_est1 = T.H @ c_est1
        est_err_coeffs1 = min_err_coeffs(z, z_est1, kvals)
        errs[idx, 0] = est_err_coeffs1[0]
        costs[idx, 0] = X_est1.fun
        
        c_est2 = X_est2.x[1:]
        z_est2 = T.H @ c_est2
        est_err_coeffs2 = min_err_coeffs(z, z_est2, kvals)
        errs[idx, 1] = est_err_coeffs2[0]
        costs[idx, 1] = X_est2.fun
        
        c_est3 = X_est3.x[1:]
        z_est3 = T.H @ c_est3
        est_err_coeffs3 = min_err_coeffs(z, z_est3, kvals)
        errs[idx, 2] = est_err_coeffs3[0]
        costs[idx, 2] = X_est3.fun
    return errs, costs

def calc_err_size_nopsftsf(L, ne, N, sizes, sd):
    # Calculation of estimation error in estimating a specific target image, multiple micrograph sizes. For the case of assuming that the measurement is well-separated.
    np.random.seed(sd)
    errs = np.zeros((len(sizes), 3))
    costs = np.zeros((len(sizes), 3))
    X_ests = np.zeros((len(sizes), 3, ne+1))
    X = np.random.rand(L, L)
    X = X / np.linalg.norm(X)
    
    W = L # L for arbitrary spacing distribution, 2*L-1 for well-separated

    B, z, roots, kvals, nu = expand_fb(X, ne)
    T = calcT(nu, kvals)
    c = np.real(T @ z)
    z = T.H@c
    Bk = np.zeros((2*L-1, 2*L-1, nu), dtype=np.complex_)
    for ii in range(nu):
        Bk[ :, :, ii] = np.fft.fft2(np.pad(np.reshape(B[ :, ii], (L, L)), L//2))


    gamma_initial = 0.09
    # y_init, s_init, locs_init = generate_clean_micrograph_2d_rots(c, kvals, Bk, W, L, N, gamma*(N/L)**2, T)
   
    X_initial1 = np.random.rand(L, L)
    X_initial1 = X_initial1 / np.linalg.norm(X_initial1)
    
    _, z_initial1, _, _, _ = expand_fb(X_initial1, ne)
    c_initial1 = np.real(T @ z_initial1)

    X_initial2 = np.random.rand(L, L)
    X_initial2 = X_initial2 / np.linalg.norm(X_initial2)
    
    _, z_initial2, _, _, _ = expand_fb(X_initial2, ne)
    c_initial2 = np.real(T @ z_initial2)
    
    X_initial3 = np.random.rand(L, L)
    X_initial3 = X_initial3 / np.linalg.norm(X_initial3)
    
    _, z_initial3, _, _, _ = expand_fb(X_initial3, ne)
    c_initial3 = np.real(T @ z_initial3)

    for (idx, sz) in enumerate(sizes):
        y_clean, s, locs = generate_clean_micrograph_2d_rots(c, kvals, Bk, W, L, sz, 0.100*(sz/L)**2, T)
        # gamma = s[0]*(L/N)**2
        sigma2 = 0
        y = y_clean + np.random.default_rng().normal(loc=0, scale=np.sqrt(sigma2), size=np.shape(y_clean))
        
        psf = np.zeros((4*L - 3, 4*L - 3))
        tsf = np.zeros((4*L - 3, 4*L - 3, 4*L - 3, 4*L - 3))
        
        ExtraMat2, ExtraMat3 = makeExtraMat(L, psf)
        tsfMat = maketsfMat(L, tsf)
        yy = np.zeros((sz, sz, 1))
        yy[ :, :, 0] = y
        
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
    
        X_est1 = Utils.optimization_funcs_rot.optimize_2d_known_psf_triplets(np.concatenate((np.reshape(gamma_initial, (1,)), c_initial1)), Bk, T, kvals, M1_y, M2_y, M3_y, sigma2, psf, L, 1, tsfMat, ExtraMat2, ExtraMat3) 
        c_est1 = X_est1.x[1:]
        z_est1 = T.H @ c_est1
        est_err_coeffs1 = min_err_coeffs(z, z_est1, kvals)
        errs[idx, 0] = est_err_coeffs1[0]
        costs[idx, 0] = X_est1.fun
        X_ests[idx, 0, :] = X_est1.x
        
        X_est2 = Utils.optimization_funcs_rot.optimize_2d_known_psf_triplets(np.concatenate((np.reshape(gamma_initial, (1,)), c_initial2)), Bk, T, kvals, M1_y, M2_y, M3_y, sigma2, psf, L, 1, tsfMat, ExtraMat2, ExtraMat3) 
        c_est2 = X_est2.x[1:]
        z_est2 = T.H @ c_est2
        est_err_coeffs2 = min_err_coeffs(z, z_est2, kvals)
        errs[idx, 1] = est_err_coeffs2[0]
        costs[idx, 1] = X_est2.fun
        X_ests[idx, 1, :] = X_est2.x
        
        X_est3 = Utils.optimization_funcs_rot.optimize_2d_known_psf_triplets(np.concatenate((np.reshape(gamma_initial, (1,)), c_initial3)), Bk, T, kvals, M1_y, M2_y, M3_y, sigma2, psf, L, 1, tsfMat, ExtraMat2, ExtraMat3) 
        c_est3 = X_est3.x[1:]
        z_est3 = T.H @ c_est3
        est_err_coeffs3 = min_err_coeffs(z, z_est3, kvals)
        errs[idx, 2] = est_err_coeffs3[0]
        costs[idx, 2] = X_est3.fun
        X_ests[idx, 2, :] = X_est3.x
    return errs, costs, X_ests, c
    