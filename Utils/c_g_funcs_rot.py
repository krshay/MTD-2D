# -*- coding: utf-8 -*-
"""
Created on Sat Jul 25 15:46:42 2020

@author: kreym
"""

import numpy as np
import scipy.special as special

import time

import multiprocessing as mp

import functools

from Utils.makeExtraMat import makeExtraMat

from Utils.funcs_calc_moments_rot import calcS3_x_gradnew2, calcS3_x_neigh_gradnew2, calcS3_x_triplets_gradnew2,  calcS2_x_grad_notparallel, calcS2_x_neigh_grad_notparallel, calck1,  calcN_mat, calcmap3, calcS3_x_grad_neigh_triplets_parallel, calcS2_x_grad_notparallel, calcS2_x_neigh_grad_notparallel
from Utils.psf_functions_2d import fourier_bessel_expansion, evaluate_psf_full

def calc_acs_grads_rot_parallel(Bk, z, kvals, L, k1_map, map3):
    # Calclulations of all needed autocorrelations and gradients, utilizing parallel processing
    kmax = np.max(kvals)
    Nmax = 6*kmax
    S3_x, gS3_x, S3_x_neigh, gS3_x_neigh, S3_x_triplets, gS3_x_triplets = calcS3_x_grad_neigh_triplets_parallel(L, Nmax, Bk, z, kvals, map3)
            
    Nmax = 4*kmax
    S2_x, gS2_x = calcS2_x_grad_notparallel(L, Nmax, Bk, z, kvals, k1_map)
    
    S2_x_neigh, gS2_x_neigh = calcS2_x_neigh_grad_notparallel(L, Bk, z, kvals, k1_map)

    return S2_x, gS2_x, S2_x_neigh, gS2_x_neigh, S3_x, gS3_x, S3_x_neigh, gS3_x_neigh, S3_x_triplets, gS3_x_triplets

def calc_acs_grads_rot_notparallel(Bk, z, kvals, L, k1_map=None, map3=None):
    # Calclulations of all needed autocorrelations and gradients, without utilizing parallel processing (faster for smaller images)
  
    kmax = np.max(kvals)
    Nmax = 6*kmax
    S3_x, gS3_x = calcS3_x_gradnew2(L, Nmax, Bk, z, kvals, map3)
    
    S3_x_neigh, gS3_x_neigh = calcS3_x_neigh_gradnew2(L, Nmax, Bk, z, kvals, map3)
    
    S3_x_triplets, gS3_x_triplets = calcS3_x_triplets_gradnew2(L, Nmax, Bk, z, kvals, map3)
    
    if k1_map == None:
        k1_map = calck1(L)
        
    Nmax = 4*kmax
    S2_x, gS2_x = calcS2_x_grad_notparallel(L, Nmax, Bk, z, kvals, k1_map)
    
    S2_x_neigh, gS2_x_neigh = calcS2_x_neigh_grad_notparallel(L, Bk, z, kvals, k1_map)
    
    return S2_x, gS2_x, S2_x_neigh, gS2_x_neigh, S3_x, gS3_x, S3_x_neigh, gS3_x_neigh, S3_x_triplets, gS3_x_triplets

def cost_grad_fun_rot_parallel(Z, Bk, T, kvals, M1_y, M2_y, M3_y, sigma2, ExtraMat2, ExtraMat3, tsfMat, L, K, N_mat=None, k1_map=None, map3=None):
    start = time.time()
    gamma = Z[:K]
    c = Z[K:]
    z = T.H@c
    
    S2_x, gS2_x, S2_x_neigh, gS2_x_neigh, S3_x, gS3_x, S3_x_neigh, gS3_x_neigh, S3_x_triplets, gS3_x_triplets = calc_acs_grads_rot_parallel(Bk, z, kvals, L, k1_map, map3)
    w1 = 1/2 
    w2 = 1/(2*L**2)
    w3 = 1/(2*L**4)

    # %% First-order moment
    S1 = np.real(np.sum(np.fft.ifftn(Bk @ z), axis=(0, 1))/(L**2))
    gS1 = np.real(np.sum(np.fft.ifftn(Bk, axes=(0,1)), axis=(0, 1))/(L**2) @ T.H)
    R1 = gamma*S1 - M1_y
    
    # %% Second-order moment
    S2 = np.real(S2_x[ :L, :L] + np.reshape(ExtraMat2*S2_x_neigh.flatten(), (L, L)))
    gS2 = np.real((np.reshape(gS2_x[ :L, :L, :], (L**2, len(z))) + ExtraMat2*np.reshape(gS2_x_neigh, ((2*L-1)**2, len(z)))) @ T.H)
    R2 = gamma*S2 - M2_y

    # Noise
    R2[0, 0] += sigma2

    # %% Third-order moment
    S3 = np.real(S3_x[ :L, :L, :L, :L] + np.reshape(ExtraMat3*S3_x_neigh.flatten(), (L, L, L, L)) + np.reshape(tsfMat*S3_x_triplets.flatten(), (L, L, L, L)))
    gS3 = np.real((np.reshape(gS3_x[ :L, :L, :L, :L, :], (L**4, len(z))) + ExtraMat3*np.reshape(gS3_x_neigh, ((2*L-1)**4, len(z)))  + tsfMat*np.reshape(gS3_x_triplets, ((2*L-1)**4, len(z)))) @ T.H)
    
    # Noise
    S3 += S1 * sigma2 * np.reshape(N_mat.toarray(), np.shape(S3))
    gS3 += N_mat @ np.reshape(gS1, (1, len(z))) * sigma2

    R3 = gamma*S3 - M3_y
    
    # %% cost and grad functions calculation
    f1 = w1*R1**2 
    f2 = w2*np.sum(R2**2) 
    f3 = w3*np.sum(R3**2)
    f = f1 + f2 + f3
    
    g_c1 = 2*w1*gamma*gS1*R1
    g_c2 = 2*w2*gamma*R2.flatten()@gS2
    g_c3 = 2*w3*gamma*R3.flatten()@gS3
    g_c = g_c1 + g_c2 + g_c3

    g_gamma1 = 2*w1*S1*R1
    g_gamma2 = 2*w2*np.sum(S2*R2)
    g_gamma3 = 2*w3*np.sum(S3*R3)
    g_gamma = g_gamma1 + g_gamma2 + g_gamma3
    
    print(f'Objective function value is {f}. gamma is {gamma}')
    print(f'Function evaluation took {time.time() - start} secs')
      
    return f, np.concatenate((np.reshape(g_gamma, (K,)), g_c))

def cost_grad_fun_rot_notparallel(Z, Bk, T, kvals, M1_y, M2_y, M3_y, sigma2, ExtraMat2, ExtraMat3, tsfMat, L, K, N_mat=None, k1_map=None, map3=None):
    gamma = Z[:K]
    c = Z[K:]
    z = T.H@c

    S2_x, gS2_x, S2_x_neigh, gS2_x_neigh, S3_x, gS3_x, S3_x_neigh, gS3_x_neigh, S3_x_triplets, gS3_x_triplets = calc_acs_grads_rot_notparallel(Bk, z, kvals, L, k1_map, map3)
    w1 = 1/2 
    w2 = 1/(2*L**2)
    w3 = 1/(2*L**4)

    # %% First-order moment
    S1 = np.real(np.sum(np.fft.ifftn(Bk @ z), axis=(0, 1))/(L**2))
    gS1 = np.real(np.sum(np.fft.ifftn(Bk, axes=(0,1)), axis=(0, 1))/(L**2) @ T.H)
    R1 = gamma*S1 - M1_y
    
    # %% Second-order moment
    S2 = np.real(S2_x[ :L, :L] + np.reshape(ExtraMat2*S2_x_neigh.flatten(), (L, L)))
    gS2 = np.real((np.reshape(gS2_x[ :L, :L, :], (L**2, len(z))) + ExtraMat2*np.reshape(gS2_x_neigh, ((2*L-1)**2, len(z)))) @ T.H)
    R2 = gamma*S2 - M2_y

    # Noise
    R2[0, 0] += sigma2

    # %% Third-order moment
    S3 = np.real(S3_x[ :L, :L, :L, :L] + np.reshape(ExtraMat3*S3_x_neigh.flatten(), (L, L, L, L)) + np.reshape(tsfMat*S3_x_triplets.flatten(), (L, L, L, L)))
    gS3 = np.real((np.reshape(gS3_x[ :L, :L, :L, :L, :], (L**4, len(z))) + ExtraMat3*np.reshape(gS3_x_neigh, ((2*L-1)**4, len(z)))  + tsfMat*np.reshape(gS3_x_triplets, ((2*L-1)**4, len(z)))) @ T.H)
    
    # Noise
    S3 += S1 * sigma2 * np.reshape(N_mat.toarray(), np.shape(S3))
    gS3 += N_mat @ np.reshape(gS1, (1, len(z))) * sigma2

    R3 = gamma*S3 - M3_y
    
    # %% cost and grad functions calculation
    f1 = w1*R1**2 
    f2 = w2*np.sum(R2**2) 
    f3 = w3*np.sum(R3**2)
    f = f1 + f2 + f3
    
    g_c1 = 2*w1*gamma*gS1*R1
    g_c2 = 2*w2*gamma*R2.flatten()@gS2
    g_c3 = 2*w3*gamma*R3.flatten()@gS3
    g_c = g_c1 + g_c2 + g_c3

    g_gamma1 = 2*w1*S1*R1
    g_gamma2 = 2*w2*np.sum(S2*R2)
    g_gamma3 = 2*w3*np.sum(S3*R3)
    g_gamma = g_gamma1 + g_gamma2 + g_gamma3
    
    return f, np.concatenate((np.reshape(g_gamma, (K,)), g_c))
