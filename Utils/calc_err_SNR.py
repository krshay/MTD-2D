# -*- coding: utf-8 -*-
"""
Created on Sun Jan 24 18:38:45 2021

@author: Shay Kreymer
"""

import numpy as np
from Utils.fb_funcs import expand_fb, min_err_coeffs, calcT
from Utils.generate_clean_micrograph_2d import generate_clean_micrograph_2d_rots
from Utils.funcs_calc_moments import M2_2d, M3_2d
from Utils.psf_tsf_funcs import full_psf_2d, full_tsf_2d, makeExtraMat, maketsfMat
import Utils.optimization_funcs_rot

def calc_err_SNR_bothcases(L, ne, N, SNRs, sd):
    """ Calculate estimation error in estimating a specific target image, multiple SNRs. For both cases: known PSF and TSF, and Algorithm 1.

    Args:
        L: diameter of the target image
        ne: number of expansion coefficients
        N: the size of the micrographs to be generated
        SNRs: an array containing the desired values of SNR
        sd: a seed

    Returns:
        errs: an array containing the estimation errors for each size (3 initial guesses)
        errs: an array containing the objective function values for each size (3 initial guesses)
    """
    # %% preliminary definitions
    np.random.seed(sd)
    errs_known = np.zeros((len(SNRs), 3))
    costs_known = np.zeros((len(SNRs), 3))
    errs_Algorithm1 = np.zeros((len(SNRs), 3))
    costs_Algorithm1 = np.zeros((len(SNRs), 3))
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

    Xrec = np.reshape(np.real(B @ z), np.shape(X))
    Xrec = Xrec / np.linalg.norm(Xrec)
    B, z, roots, kvals, nu = expand_fb(Xrec, ne)
    T = calcT(nu, kvals)
    c = np.real(T @ z)
    z = T.H@c
    Bk = np.zeros((2*L-1, 2*L-1, nu), dtype=np.complex_)
    for ii in range(nu):
        Bk[ :, :, ii] = np.fft.fft2(np.pad(np.reshape(B[ :, ii], (L, L)), L//2))
    Xrec = np.reshape(np.real(B @ z), np.shape(X))
    
    # %% initial guesses
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
    
    # %% calculations
    y_clean, s, locs = generate_clean_micrograph_2d_rots(c, kvals, Bk, W, L, N, 0.100*(N/L)**2, T, seed=sd)
    psf = full_psf_2d(locs, L)
    tsf = full_tsf_2d(locs, L)
    
    ExtraMat2, ExtraMat3 = makeExtraMat(L, psf)
    tsfMat = maketsfMat(L, tsf)
    for (idx, SNR) in enumerate(SNRs):
        sigma2 = np.linalg.norm(Xrec)**2 / (SNR * np.pi * (L//2)**2)
        y = y_clean + np.random.default_rng().normal(loc=0, scale=np.sqrt(sigma2), size=np.shape(y_clean))
        
        yy = np.zeros((N, N, 1))
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
        del y
        X1_known = Utils.optimization_funcs_rot.optimize_2d_known_psf_triplets(np.concatenate((np.reshape(gamma_initial, (1,)), c_initial1)), Bk, T, kvals, M1_y, M2_y, M3_y, sigma2, L, 1, tsfMat, ExtraMat2, ExtraMat3) 
        X2_known = Utils.optimization_funcs_rot.optimize_2d_known_psf_triplets(np.concatenate((np.reshape(gamma_initial, (1,)), c_initial2)), Bk, T, kvals, M1_y, M2_y, M3_y, sigma2, L, 1, tsfMat, ExtraMat2, ExtraMat3) 
        X3_known = Utils.optimization_funcs_rot.optimize_2d_known_psf_triplets(np.concatenate((np.reshape(gamma_initial, (1,)), c_initial3)), Bk, T, kvals, M1_y, M2_y, M3_y, sigma2, L, 1, tsfMat, ExtraMat2, ExtraMat3) 
        
        c_est1 = X1_known.x[1:]
        z_est1 = T.H @ c_est1
        est_err_coeffs1 = min_err_coeffs(z, z_est1, kvals)
        errs_known[idx, 0] = est_err_coeffs1[0]
        costs_known[idx, 0] = X1_known.fun
        
        c_est2 = X2_known.x[1:]
        z_est2 = T.H @ c_est2
        est_err_coeffs2 = min_err_coeffs(z, z_est2, kvals)
        errs_known[idx, 1] = est_err_coeffs2[0]
        costs_known[idx, 1] = X2_known.fun
        
        c_est3 = X3_known.x[1:]
        z_est3 = T.H @ c_est3
        est_err_coeffs3 = min_err_coeffs(z, z_est3, kvals)
        errs_known[idx, 2] = est_err_coeffs3[0]
        costs_known[idx, 2] = X3_known.fun
        
        X_Algorithm1_1, _, _ = Utils.optimization_funcs_rot.optimize_rot_Algorithm1_notparallel(np.concatenate((np.reshape(gamma_initial, (1,)), c_initial1)), Bk, T, kvals, M1_y, M2_y, M3_y, sigma2, L, 1, W, 10000, iters_till_change=150) 
        X_Algorithm1_2, _, _ = Utils.optimization_funcs_rot.optimize_rot_Algorithm1_notparallel(np.concatenate((np.reshape(gamma_initial, (1,)), c_initial2)), Bk, T, kvals, M1_y, M2_y, M3_y, sigma2, L, 1, W, 10000, iters_till_change=150) 
        X_Algorithm1_3, _, _ = Utils.optimization_funcs_rot.optimize_rot_Algorithm1_notparallel(np.concatenate((np.reshape(gamma_initial, (1,)), c_initial3)), Bk, T, kvals, M1_y, M2_y, M3_y, sigma2, L, 1, W, 10000, iters_till_change=150) 

        c_est1 = X_Algorithm1_1.x[1:]
        z_est1 = T.H @ c_est1
        est_err_coeffs1 = min_err_coeffs(z, z_est1, kvals)
        errs_Algorithm1[idx, 0] = est_err_coeffs1[0]
        costs_Algorithm1[idx, 0] = X_Algorithm1_1.fun
        
        c_est2 = X_Algorithm1_2.x[1:]
        z_est2 = T.H @ c_est2
        est_err_coeffs2 = min_err_coeffs(z, z_est2, kvals)
        errs_Algorithm1[idx, 1] = est_err_coeffs2[0]
        costs_Algorithm1[idx, 1] = X_Algorithm1_2.fun
        
        c_est3 = X_Algorithm1_3.x[1:]
        z_est3 = T.H @ c_est3
        est_err_coeffs3 = min_err_coeffs(z, z_est3, kvals)
        errs_Algorithm1[idx, 2] = est_err_coeffs3[0]
        costs_Algorithm1[idx, 2] = X_Algorithm1_3.fun
        
    return errs_known, costs_known, errs_Algorithm1, costs_Algorithm1
