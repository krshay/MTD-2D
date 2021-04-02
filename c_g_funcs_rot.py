# -*- coding: utf-8 -*-
"""
Created on Sat Jul 25 15:46:42 2020

@author: kreym
"""

import numpy as np
import scipy.special as special

import time

# import torch

import multiprocessing as mp

import functools

from makeExtraMat import makeExtraMat

# import time

from funcs_calc_moments_rot import calcS3_x_gradnew2, calcS3_x_neigh_gradnew2, calcS3_x_triplets_gradnew2, calcS3_x_grad_binned, calcS3_x_neigh_grad_binned, calck1k2k3, calcS3_grad_full_shift, calcS2_x_grad, calcS2_x_neigh_grad, calck1, calcS2_grad_full_shift, calck1, calcS2_grad_full_shift, calcN_mat, calcS3_x_grad_parallel, calcS2_grad_full_shift_coeffs, calcS3_grad_full_shift_coeffs, calcS3_x_grad_k, calcS2_x_grad_k, calc_gpsf, calcmap3, calcS3_x_grad_neigh_triplets_parallel
from funcs_calc_lists import calclist2_all, calclist3_all
from psf_functions_2d import fourier_bessel_expansion, evaluate_psf_full

def calc_acs_grads_rot(Bk, z, kvals, L, k1_map=None, k1k2k3_map=None):
    # Calclulations of all needed autocorrelations and gradients
    if k1k2k3_map == None:
        k1k2k3_map = calck1k2k3(L)
    
    kmax = np.max(kvals)
    Nmax = 6*kmax
    S3_x, gS3_x = calcS3_x_gradnew2(L, Nmax, Bk, z, kvals, k1k2k3_map)
    
    Nmax_neigh = 4*kmax
    S3_x_neigh, gS3_x_neigh = calcS3_x_neigh_gradnew2(L, Nmax_neigh, Bk, z, kvals, k1k2k3_map)
    
    if k1_map == None:
        k1_map = calck1(L)
        
    Nmax = 4*kmax
    S2_x, gS2_x = calcS2_x_grad(L, Nmax, Bk, z, kvals, k1_map)
    
    S2_x_neigh, gS2_x_neigh = calcS2_x_neigh_grad(L, Bk, z, kvals, k1_map)
    
    return S2_x, gS2_x, S2_x_neigh, gS2_x_neigh, S3_x, gS3_x, S3_x_neigh, gS3_x_neigh

def calc_acs_grads_rot_parallel(Bk, z, kvals, L, k1_map=None, k1k2k3_map=None):
    # Calclulations of all needed autocorrelations and gradients, utilizing parallel processing
    if k1k2k3_map == None:
        k1k2k3_map = calck1k2k3(L)
    kmax = np.max(kvals)
    Nmax = 6*kmax
    S3_x, gS3_x, S3_x_neigh, gS3_x_neigh = calcS3_x_grad_parallel(L, Nmax, Bk, z, kvals, k1k2k3_map)
    
    if k1_map == None:
        k1_map = calck1(L)
        
    Nmax = 4*kmax
    S2_x, gS2_x = calcS2_x_grad(L, Nmax, Bk, z, kvals, k1_map)
    
    S2_x_neigh, gS2_x_neigh = calcS2_x_neigh_grad(L, Bk, z, kvals, k1_map)

    return S2_x, gS2_x, S2_x_neigh, gS2_x_neigh, S3_x, gS3_x, S3_x_neigh, gS3_x_neigh

def calc_acs_grads_rot_triplets_parallel(Bk, z, kvals, L, k1_map, map3):
    # Calclulations of all needed autocorrelations and gradients, utilizing parallel processing
    kmax = np.max(kvals)
    Nmax = 6*kmax
    S3_x, gS3_x, S3_x_neigh, gS3_x_neigh, S3_x_triplets, gS3_x_triplets = calcS3_x_grad_neigh_triplets_parallel(L, Nmax, Bk, z, kvals, map3)
            
    Nmax = 4*kmax
    S2_x, gS2_x = calcS2_x_grad(L, Nmax, Bk, z, kvals, k1_map)
    
    S2_x_neigh, gS2_x_neigh = calcS2_x_neigh_grad(L, Bk, z, kvals, k1_map)

    return S2_x, gS2_x, S2_x_neigh, gS2_x_neigh, S3_x, gS3_x, S3_x_neigh, gS3_x_neigh, S3_x_triplets, gS3_x_triplets

def calc_acs_grads_rot_triplets(Bk, z, kvals, L, k1_map=None, map3=None):
    # Calclulations of all needed autocorrelations and gradients
  
    kmax = np.max(kvals)
    Nmax = 6*kmax
    S3_x, gS3_x = calcS3_x_gradnew2(L, Nmax, Bk, z, kvals, map3)
    
    S3_x_neigh, gS3_x_neigh = calcS3_x_neigh_gradnew2(L, Nmax, Bk, z, kvals, map3)
    
    S3_x_triplets, gS3_x_triplets = calcS3_x_triplets_gradnew2(L, Nmax, Bk, z, kvals, map3)
    
    if k1_map == None:
        k1_map = calck1(L)
        
    Nmax = 4*kmax
    S2_x, gS2_x = calcS2_x_grad(L, Nmax, Bk, z, kvals, k1_map)
    
    S2_x_neigh, gS2_x_neigh = calcS2_x_neigh_grad(L, Bk, z, kvals, k1_map)
    
    return S2_x, gS2_x, S2_x_neigh, gS2_x_neigh, S3_x, gS3_x, S3_x_neigh, gS3_x_neigh, S3_x_triplets, gS3_x_triplets


def calc_acs_grads_rot_binned(Bk, z, kvals, L, k1_map, k1k2k3_map_binned, k1k2k3_map_binned_idxs):
    # Calclulations of all needed autocorrelations and gradients utilizing binning
    kmax = np.max(kvals)
    Nmax = 6*kmax
    S3_x_binned, gS3_x_binned = calcS3_x_grad_binned(L, Nmax, Bk, z, kvals, k1k2k3_map_binned, k1k2k3_map_binned_idxs)
    
    Nmax_neigh = 4*kmax
    S3_x_neigh_binned, gS3_x_neigh_binned = calcS3_x_neigh_grad_binned(L, Nmax_neigh, Bk, z, kvals, k1k2k3_map_binned, k1k2k3_map_binned_idxs)
    
    Nmax = 4*kmax
    S2_x, gS2_x = calcS2_x_grad(L, Nmax, Bk, z, kvals, k1_map)
    
    S2_x_neigh, gS2_x_neigh = calcS2_x_neigh_grad(L, Bk, z, kvals, k1_map)
    
    return S2_x, gS2_x, S2_x_neigh, gS2_x_neigh, S3_x_binned, gS3_x_binned, S3_x_neigh_binned, gS3_x_neigh_binned

def cost_grad_fun_2d_known_psf_mat_rot_gamma(Z, Bk, T, kvals, M1data, M2data, M3data, sigma2, ExtraMat2, ExtraMat3, psf, L, K, N_mat=None, k1_map=None, k1k2k3_map=None):
    gamma = Z[:K]###### NOTICE for K > 1
    c = Z[K:] ###### NOTICE for K > 1
    z = T.H@c
    if k1k2k3_map == None:
        k1k2k3_map = calck1k2k3(L)
    if k1_map == None:
        k1_map = calck1(L)
    if N_mat is None:
        N_mat = calcN_mat(L)

    S2_x, gS2_x, S2_x_neigh, gS2_x_neigh, S3_x, gS3_x, S3_x_neigh, gS3_x_neigh = calc_acs_grads_rot_parallel(Bk, z, kvals, L, k1_map, k1k2k3_map)

    w1 = 1/2 
    w2 = 1/(2*L**2)
    w3 = 1/(2*L**4)

    # %% First-order moment, forward model
    S1 = np.real(np.sum(np.fft.ifftn(Bk @ z), axis=(0, 1))/(L**2))
    gS1 = np.real(np.sum(np.fft.ifftn(Bk, axes=(0,1)), axis=(0, 1))/(L**2) @ T.H)
    R1 = gamma*S1 - M1data
    
    # %% Second-order moment, forward model
    S2 = np.real(S2_x[ :L, :L] + np.reshape(ExtraMat2*S2_x_neigh.flatten(), (L, L)))
    gS2 = np.real((np.reshape(gS2_x[ :L, :L, :], (L**2, len(z))) + ExtraMat2*np.reshape(gS2_x_neigh, ((2*L-1)**2, len(z)))) @ T.H)
    R2 = gamma*S2 - M2data

    # Noise
    R2[0, 0] += sigma2

    # %% Third-order moment, forward model
    S3 = np.real(S3_x[ :L, :L, :L, :L] + np.reshape(ExtraMat3*S3_x_neigh.flatten(), (L, L, L, L)))
    gS3 = np.real((np.reshape(gS3_x[ :L, :L, :L, :L, :], (L**4, len(z))) + ExtraMat3*np.reshape(gS3_x_neigh, ((2*L-1)**4, len(z)))) @ T.H)
    
    # Noise
    S3 += S1 * sigma2 * np.reshape(N_mat.toarray(), np.shape(S3))
    gS3 += N_mat @ np.reshape(gS1, (1, len(z))) * sigma2

    R3 = gamma*S3 - M3data
    
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
    
    # print(f)
    # print(gamma)
    # print(c)
    # print(f'Time passed for a regular function evaluation: {time.time() - t1}')
    return f, np.concatenate((np.reshape(g_gamma, (K,)), g_c)) # 


def cost_grad_fun_2d_known_psf_triplets(Z, Bk, T, kvals, M1data, M2data, M3data, sigma2, ExtraMat2, ExtraMat3, tsfMat, L, K, N_mat=None, k1_map=None, map3=None):
    gamma = Z[:K]###### NOTICE for K > 1
    c = Z[K:] ###### NOTICE for K > 1
    z = T.H@c
    if k1_map == None:
        k1_map = calck1(L)
    if N_mat is None:
        N_mat = calcN_mat(L)

    S2_x, gS2_x, S2_x_neigh, gS2_x_neigh, S3_x, gS3_x, S3_x_neigh, gS3_x_neigh, S3_x_triplets, gS3_x_triplets = calc_acs_grads_rot_triplets(Bk, z, kvals, L, k1_map, map3)

    w1 = 1/2 
    w2 = 1/(2*L**2)
    w3 = 1/(2*L**4)

    # %% First-order moment, forward model
    S1 = np.real(np.sum(np.fft.ifftn(Bk @ z), axis=(0, 1))/(L**2))
    gS1 = np.real(np.sum(np.fft.ifftn(Bk, axes=(0,1)), axis=(0, 1))/(L**2) @ T.H)
    R1 = gamma*S1 - M1data
    
    # %% Second-order moment, forward model
    S2 = np.real(S2_x[ :L, :L] + np.reshape(ExtraMat2*S2_x_neigh.flatten(), (L, L)))
    gS2 = np.real((np.reshape(gS2_x[ :L, :L, :], (L**2, len(z))) + ExtraMat2*np.reshape(gS2_x_neigh, ((2*L-1)**2, len(z)))) @ T.H)
    R2 = gamma*S2 - M2data

    # Noise
    R2[0, 0] += sigma2

    # %% Third-order moment, forward model
    S3 = np.real(S3_x[ :L, :L, :L, :L] + np.reshape(ExtraMat3*S3_x_neigh.flatten(), (L, L, L, L)) + np.reshape(tsfMat*S3_x_triplets.flatten(), (L, L, L, L)))
    gS3 = np.real((np.reshape(gS3_x[ :L, :L, :L, :L, :], (L**4, len(z))) + ExtraMat3*np.reshape(gS3_x_neigh, ((2*L-1)**4, len(z)))  + tsfMat*np.reshape(gS3_x_triplets, ((2*L-1)**4, len(z)))) @ T.H)
    
    # Noise
    S3 += S1 * sigma2 * np.reshape(N_mat.toarray(), np.shape(S3))
    gS3 += N_mat @ np.reshape(gS1, (1, len(z))) * sigma2

    R3 = gamma*S3 - M3data
    
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
    
    # print(f)
    # print(gamma)
    # print(c)
    # print(f'Time passed for a regular function evaluation: {time.time() - t1}')
    return f, np.concatenate((np.reshape(g_gamma, (K,)), g_c)) # 

def cost_grad_fun_2d_known_psf_tripletsnew(Z, Bk, T, kvals, M1data, M2data, M3data, sigma2, ExtraMat2, ExtraMat3, tsfMat, L, K, N_mat=None, k1_map=None, map3=None):
    gamma = Z[:K]###### NOTICE for K > 1
    c = Z[K:] ###### NOTICE for K > 1
    z = T.H@c
    
    S2_x, gS2_x, S2_x_neigh, gS2_x_neigh, S3_x, gS3_x, S3_x_neigh, gS3_x_neigh, S3_x_triplets, gS3_x_triplets = calc_acs_grads_rot_triplets_parallel(Bk, z, kvals, L, k1_map, map3)
    w1 = 1/2 
    w2 = 1/(2*L**2)
    w3 = 1/(2*L**4)

    # %% First-order moment, forward model
    S1 = np.real(np.sum(np.fft.ifftn(Bk @ z), axis=(0, 1))/(L**2))
    gS1 = np.real(np.sum(np.fft.ifftn(Bk, axes=(0,1)), axis=(0, 1))/(L**2) @ T.H)
    R1 = gamma*S1 - M1data
    
    # %% Second-order moment, forward model
    S2 = np.real(S2_x[ :L, :L] + np.reshape(ExtraMat2*S2_x_neigh.flatten(), (L, L)))
    gS2 = np.real((np.reshape(gS2_x[ :L, :L, :], (L**2, len(z))) + ExtraMat2*np.reshape(gS2_x_neigh, ((2*L-1)**2, len(z)))) @ T.H)
    R2 = gamma*S2 - M2data

    # Noise
    R2[0, 0] += sigma2

    # %% Third-order moment, forward model
    S3 = np.real(S3_x[ :L, :L, :L, :L] + np.reshape(ExtraMat3*S3_x_neigh.flatten(), (L, L, L, L)) + np.reshape(tsfMat*S3_x_triplets.flatten(), (L, L, L, L)))
    gS3 = np.real((np.reshape(gS3_x[ :L, :L, :L, :L, :], (L**4, len(z))) + ExtraMat3*np.reshape(gS3_x_neigh, ((2*L-1)**4, len(z)))  + tsfMat*np.reshape(gS3_x_triplets, ((2*L-1)**4, len(z)))) @ T.H)
    
    # Noise
    S3 += S1 * sigma2 * np.reshape(N_mat.toarray(), np.shape(S3))
    gS3 += N_mat @ np.reshape(gS1, (1, len(z))) * sigma2

    R3 = gamma*S3 - M3data
    
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
    
    print(f)
    print(gamma)
    
    return f, np.concatenate((np.reshape(g_gamma, (K,)), g_c))

def cost_grad_fun_2d_known_psf_tripletsnotparallelnew(Z, Bk, T, kvals, M1data, M2data, M3data, sigma2, ExtraMat2, ExtraMat3, tsfMat, L, K, N_mat=None, k1_map=None, map3=None):
    gamma = Z[:K]###### NOTICE for K > 1
    c = Z[K:] ###### NOTICE for K > 1
    z = T.H@c

    S2_x, gS2_x, S2_x_neigh, gS2_x_neigh, S3_x, gS3_x, S3_x_neigh, gS3_x_neigh, S3_x_triplets, gS3_x_triplets = calc_acs_grads_rot_triplets(Bk, z, kvals, L, k1_map, map3)
    w1 = 1/2 
    w2 = 1/(2*L**2)
    w3 = 1/(2*L**4)

    # %% First-order moment, forward model
    S1 = np.real(np.sum(np.fft.ifftn(Bk @ z), axis=(0, 1))/(L**2))
    gS1 = np.real(np.sum(np.fft.ifftn(Bk, axes=(0,1)), axis=(0, 1))/(L**2) @ T.H)
    R1 = gamma*S1 - M1data
    
    # %% Second-order moment, forward model
    S2 = np.real(S2_x[ :L, :L] + np.reshape(ExtraMat2*S2_x_neigh.flatten(), (L, L)))
    gS2 = np.real((np.reshape(gS2_x[ :L, :L, :], (L**2, len(z))) + ExtraMat2*np.reshape(gS2_x_neigh, ((2*L-1)**2, len(z)))) @ T.H)
    R2 = gamma*S2 - M2data

    # Noise
    R2[0, 0] += sigma2

    # %% Third-order moment, forward model
    S3 = np.real(S3_x[ :L, :L, :L, :L] + np.reshape(ExtraMat3*S3_x_neigh.flatten(), (L, L, L, L)) + np.reshape(tsfMat*S3_x_triplets.flatten(), (L, L, L, L)))
    gS3 = np.real((np.reshape(gS3_x[ :L, :L, :L, :L, :], (L**4, len(z))) + ExtraMat3*np.reshape(gS3_x_neigh, ((2*L-1)**4, len(z)))  + tsfMat*np.reshape(gS3_x_triplets, ((2*L-1)**4, len(z)))) @ T.H)
    
    # Noise
    S3 += S1 * sigma2 * np.reshape(N_mat.toarray(), np.shape(S3))
    gS3 += N_mat @ np.reshape(gS1, (1, len(z))) * sigma2

    R3 = gamma*S3 - M3data
    
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
    
    print(f)
    # print(gamma)
    # print(c)
    # print(f'Time passed for a regular function evaluation: {time.time() - t1}')
    return f, np.concatenate((np.reshape(g_gamma, (K,)), g_c)) # 


def cost_grad_fun_2d_known_psf_binnednotparallel(Z, Bk, T, kvals, M1data, M2data, M3data, sigma2, ExtraMat2, ExtraMat3, L, K, N_mat=None, k1_map=None, k1k2k3_map_binned=None, k1k2k3_map_binned_idxs=None):
    gamma = Z[:K]###### NOTICE for K > 1
    c = Z[K:] ###### NOTICE for K > 1
    z = T.H@c

    S2_x, gS2_x, S2_x_neigh, gS2_x_neigh, S3_x, gS3_x, S3_x_neigh, gS3_x_neigh = calc_acs_grads_rot_binned(Bk, z, kvals, L, k1_map, k1k2k3_map_binned, k1k2k3_map_binned_idxs)
    w1 = 1/2 
    w2 = 1/(2*L**2)
    w3 = 1/(2*L**4)

    # %% First-order moment, forward model
    S1 = np.real(np.sum(np.fft.ifftn(Bk @ z), axis=(0, 1))/(L**2))
    gS1 = np.real(np.sum(np.fft.ifftn(Bk, axes=(0,1)), axis=(0, 1))/(L**2) @ T.H)
    R1 = gamma*S1 - M1data
    
    # %% Second-order moment, forward model
    S2 = np.real(S2_x[ :L, :L] + np.reshape(ExtraMat2*S2_x_neigh.flatten(), (L, L)))
    gS2 = np.real((np.reshape(gS2_x[ :L, :L, :], (L**2, len(z))) + ExtraMat2*np.reshape(gS2_x_neigh, ((2*L-1)**2, len(z)))) @ T.H)
    R2 = gamma*S2 - M2data

    # Noise
    R2[0, 0] += sigma2

    # %% Third-order moment, forward model
    S3 = np.real(S3_x[ :L, :L, :L, :L] + np.reshape(ExtraMat3*S3_x_neigh.flatten(), (L, L, L, L)))
    gS3 = np.real((np.reshape(gS3_x[ :L, :L, :L, :L, :], (L**4, len(z))) + ExtraMat3*np.reshape(gS3_x_neigh, ((2*L-1)**4, len(z)))) @ T.H)
    
    # Noise
    S3 += S1 * sigma2 * np.reshape(N_mat.toarray(), np.shape(S3))
    gS3 += N_mat @ np.reshape(gS1, (1, len(z))) * sigma2

    R3 = gamma*S3 - M3data
    
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
    
    print(f)
    print(gamma)
    # print(c)
    # print(f'Time passed for a regular function evaluation: {time.time() - t1}')
    return f, np.concatenate((np.reshape(g_gamma, (K,)), g_c)) # 






def cost_grad_fun_2d_psf(Z, gamma, S1, S2_x, S2_x_neigh, S3_x, S3_x_neigh, Bk, kvals, M1data, M2data, M3data, sigma2, L, K, N_mat):
    psf = np.reshape(Z, (4*L-3, 4*L-3))
    
    w3 = 1/(2*L**4)

    
    # %% Second-order moment, forward model
    # S2_z = np.zeros((L, L), dtype=np.complex_)
    # gS2_z = np.zeros((L, L, len(z)), dtype=np.complex_)
    # for i1 in range(L):
    #     for j1 in range(L):
    #             S2_z[i1, j1], gS2_z[i1, j1, :] = calcS2_grad_full_shift((i1, j1), S2_x, gS2_x, S2_x_neigh, gS2_x_neigh, L, psf)    

    # S2check = np.real(S2_z[ :L, :L])
    # gS2check = np.real(np.reshape(gS2_z[ :L, :L, :], (L**2, len(z))) @ T.H)
    
    # S2 = np.real(S2_x[ :L, :L] + np.reshape(ExtraMat2*S2_x_neigh.flatten(), (L, L)))
    # gS2 = np.real((np.reshape(gS2_x[ :L, :L, :], (L**2, len(z))) + ExtraMat2*np.reshape(gS2_x_neigh, ((2*L-1)**2, len(z)))) @ T.H)
    # R2 = gamma*S2 - M2data

    # # Noise
    # R2[0, 0] += sigma2

    # %% Third-order moment, forward model
    _, ExtraMat3 = makeExtraMat(L, psf)
    S3 = np.real(S3_x[ :L, :L, :L, :L] + np.reshape(ExtraMat3*S3_x_neigh.flatten(), (L, L, L, L)))
    # Noise
    S3 += S1 * sigma2 * np.reshape(N_mat.toarray(), np.shape(S3))
    
    R3 = gamma*S3 - M3data
    
    # %% cost and grad functions calculation
    # f2 = w2*np.sum(R2**2) 
    f3 = w3*np.sum(R3**2)
    f = f3
    
    gpsf = calc_gpsf(L, S3_x_neigh)
    g_psf = 2*w3*gamma*R3.flatten()@gpsf

    
    # print(f)
    # print(gamma)
    # print(c)
    # print(f'Time passed for a regular function evaluation: {time.time() - t1}')
    return f, g_psf # 
