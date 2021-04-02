# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 11:22:42 2020

@author: kreym
"""

import numpy as np
import scipy.special as special

# import multiprocessing as mp

# import time

from funcs_calc_moments import M2_2d_ac_grad, M3_2d_ac_grad
from funcs_calc_lists import calclist2_all, calclist3_all
from psf_functions_2d import fourier_bessel_expansion, evaluate_psf_full

def calc_acs_grads(X, L, list2_all=None, list3_all=None):
    # Calclulations of all needed autocorrelations and gradients
    dict_ac = {}
    dict_grads = {}
    
    if list2_all == None:
        list2_all = calclist2_all(L)
    for shift in list2_all:
        ac_grad_2 = M2_2d_ac_grad(X, shift)
        dict_ac[shift] = ac_grad_2[0]
        dict_grads[shift] = ac_grad_2[1]
    
    if list3_all == None:
        list3_all = calclist3_all(L)
    for shift in list3_all:
        ac_grad_3 = M3_2d_ac_grad(X, shift[0], shift[1])
        dict_ac[shift] = ac_grad_3[0]
        dict_grads[shift] = ac_grad_3[1]
        # dict_ac[shift[::-1]] = ac_grad_3[0]
        # dict_grads[shift[::-1]] = ac_grad_3[1]
    
    return dict_ac, dict_grads

def calc_acs_grads_parallel(X, L, pool, list2_all=None, list3_all=None, list3_all_reversed=None):
    # Calclulations of all needed autocorrelations and gradients using parallel processing
    if list2_all == None:
        list2_all = calclist2_all(L)
    ac_grad_2 = pool.starmap(M2_2d_ac_grad, [(X, shift) for shift in list2_all])
    ac_grad_2 = list(zip(*ac_grad_2))
    dict_ac_2 = dict(zip(list2_all, ac_grad_2[0]))
    dict_grads_2 = dict(zip(list2_all, ac_grad_2[1]))

    if list3_all == None:
        list3_all = calclist3_all(L)
        # list3_all_reversed = [t[::-1] for t in list3_all]
    ac_grad_3 = pool.starmap(M3_2d_ac_grad, [(X, shift[0], shift[1]) for shift in list3_all])
    ac_grad_3 = list(zip(*ac_grad_3))
    dict_ac_3 = dict(zip(list3_all, ac_grad_3[0]))# {** #, **dict(zip(list3_all_reversed, ac_grad_3[0]))}
    dict_grads_3 = dict(zip(list3_all, ac_grad_3[1]))# {** #, **dict(zip(list3_all_reversed, ac_grad_3[1]))}
    
    return {**dict_ac_2, **dict_ac_3}, {**dict_grads_2, **dict_grads_3}

def cost_grad_fun_2d_known_psf(Z, params, psf, triplets, L, K, list2_all=None, list3_all=None, list3_all_reversed=None, pool=None):
    # %% Definitions
    M1data = params[0]
    M2data = params[1]
    M3data = params[2]
    list2 = params[3]
    list3 = params[4]
    
    if list2_all == None:
        list2_all = calclist2_all(L)
    if list3_all == None:
        list3_all = calclist3_all(L)
        list3_all_reversed = [t[::-1] for t in list3_all]
    
    gamma = np.reshape(Z[:K], (K,))
    X = np.reshape(Z[K:], (L,L,K)) ###### NOTICE for K > 1
    if pool == None:
        dict_ac, dict_grads = calc_acs_grads(X, L, list2_all, list3_all)
    else:
        dict_ac, dict_grads = calc_acs_grads_parallel(X, L, pool, list2_all, list3_all, list3_all_reversed)
    n2 = np.shape(list2)[0]
    n3 = np.shape(list3)[0]
    w1 = 1/2 
    w2 = 1/(2*n2)
    w3 = 1/(2*n3)

    # %% First-order moment, forward model
    mean_X = np.sum(X, axis=(0, 1))/(L**2)
    M1 = gamma.T @ mean_X
    R1 = M1 - M1data
    
    # if pool == None:
    # %% Second-order moments, forward model
    M2s = [calc_2k_known_psf(k, list2[k], dict_ac, dict_grads, L, K, psf, M2data[k], gamma) for k in range(n2)]

    # %% Third-order moments, forward model
    M3s = [calc_3k_known_psf(k, list3[k][0], list3[k][1], dict_ac, dict_grads, L, K, psf, M3data[k], gamma, triplets) for k in range(n3)]
    # else:
    #     # %% Second-order moments, forward model
    #     M2s = pool.starmap(calc_2k, [(k, list2[k], dict_ac, dict_grads, L, K, psf, M2data[k], gamma) for k in range(n2)])
    
    #     # %% Third-order moments, forward model
    #     M3s = pool.starmap(calc_3k, [(k, list3[k][0], list3[k][1], dict_ac, dict_grads, L, K, psf, M3data[k], gamma, triplets) for k in range(n3)])
    
    # %% Calculations of cost and gradients
    M2 = np.asarray([M[0] for M in M2s])
    R2 = np.asarray([M[1] for M in M2s])
    T2 = [M[2] for M in M2s]
    M3 = np.asarray([M[0] for M in M3s])
    R3 = np.asarray([M[1] for M in M3s])
    T3 = [M[2] for M in M3s]
    
    f = w1*R1**2 + w2*R2@R2 + w3*R3@R3
    
    R2T2 = 0
    for k in range(n2):
        R2T2 = R2T2 + R2[k]*T2[k]
    
    R3T3 = 0
    for k in range(n3):
        R3T3 = R3T3 + R3[k]*T3[k]
    
    g_X = (2*w1*R1/(L**2)) * np.ones((L, L, K)) * gamma.T \
        + 2*w2*R2T2 \
            + 2*w3*R3T3
    g_gamma = 2*w1*R1*mean_X \
        + 2*w2*R2@M2 \
            + 2*w3*R3@M3

    return f, np.concatenate((np.reshape(g_gamma, (K,)), g_X.flatten()))

def calc_2k_known_psf(k, shift1, dict_ac, dict_grads, L, K, psf, M2data_k, gamma):
    shift1y = shift1[0]
    shift1x = shift1[1]
    
    # %% 1.
    M2_clean = dict_ac[shift1]
    T2_clean = dict_grads[shift1]
    
    M2k_extra = np.zeros(K)
    T2k_extra = np.zeros((L, L, K))
    
    # %% 2.
    for j in range(shift1y-(L-1), L + shift1y):
        for i in range(shift1x-(L-1), L + shift1x):
            if not (np.abs(i) < L and np.abs(j) < L):
                M2k_extra = M2k_extra + evaluate_psf_full(psf, (j,i), L) \
                * dict_ac[(j-shift1y, i-shift1x)]#evaluate_psf(psf, np.sqrt(i**2 + j**2))
                T2k_extra = T2k_extra + evaluate_psf_full(psf, (j,i), L) \
                * dict_grads[(j-shift1y, i-shift1x)]
                
    M2k = M2_clean + M2k_extra
    R2k = (gamma.T @ M2k) - M2data_k #+ bias2[k]
    T2k = (T2_clean + T2k_extra) * gamma.T # scale column k by gamma(k)
    
    return (M2k[0], R2k, T2k)

def calc_3k_known_psf(k, shift1, shift2, dict_ac, dict_grads, L, K, psf, M3data_k, gamma, triplets):
    shift1y = shift1[0]
    shift1x = shift1[1]

    shift2y = shift2[0]
    shift2x = shift2[1]
            
    # %% 1.
    M3_clean = dict_ac[(shift1, shift2)]
    T3_clean = dict_grads[(shift1, shift2)]
    
    # %% 2. ->
    M3k_extra = np.zeros(K)
    T3k_extra = np.zeros((L, L, K))
    
    # %% 2. 
    for j in range(max(shift1y, shift2y) - (L-1), L + min(shift1y, shift2y)):
        for i in range(max(shift1x, shift2x) - (L-1), L + min(shift1x, shift2x)):
            if not (np.abs(i) < L and np.abs(j) < L):
                M3k_extra = M3k_extra + evaluate_psf_full(psf, (j,i), L) * dict_ac[((shift2y-shift1y, shift2x-shift1x), (j-shift1y, i-shift1x))]
                T3k_extra = T3k_extra + evaluate_psf_full(psf, (j,i), L) * dict_grads[((shift2y-shift1y, shift2x-shift1x), (j-shift1y, i-shift1x))]
           
    # %% 3. 
    for j in range(shift1y - (L-1), L + shift1y - shift2y):
        for i in range(shift1x - (L-1), L + shift1x - shift2x):
            if not (np.abs(i) < L and np.abs(j) < L):
                M3k_extra = M3k_extra + evaluate_psf_full(psf, (j,i), L) * dict_ac[((j-shift1y, i-shift1x), (j+shift2y-shift1y, i+shift2x-shift1x))]
                T3k_extra = T3k_extra + evaluate_psf_full(psf, (j,i), L) * dict_grads[((j-shift1y, i-shift1x), (j+shift2y-shift1y, i+shift2x-shift1x))]
    
    # %% 4. 
    for j in range(shift2y - (L-1), L + shift2y - shift1y):
        for i in range(shift2x - (L-1), L + shift2x - shift1x):
            if not (np.abs(i) < L and np.abs(j) < L):
                M3k_extra = M3k_extra + evaluate_psf_full(psf, (j,i), L) * dict_ac[((j-shift2y, i-shift2x), (j+shift1y-shift2y, i+shift1x-shift2x))]
                T3k_extra = T3k_extra + evaluate_psf_full(psf, (j,i), L) * dict_grads[((j-shift2y, i-shift2x), (j+shift1y-shift2y, i+shift1x-shift2x))]
             
    if triplets:
        # %% 5.                 
        for j1 in range(shift1y - (L-1), L + shift1y):
            for i1 in range(shift1x - (L-1), L + shift1x):
                if not (np.abs(i1) < L and np.abs(j1) < L):
                    for j2 in range(max(j1, shift1y) - (L-1) - shift2y, L + min(j1, shift1y) - shift2y):
                        for i2 in range(max(i1, shift1x) - (L-1) - shift2x, L + min(i1, shift1x) - shift2x):
                            if not (np.abs(i2) < L and np.abs(j2) < L):
                                if not (np.abs(i2 - i1) < L and np.abs(j2 - j1) < L):
                                    M3k_extra = M3k_extra + evaluate_psf_full(psf, (j1,i1), L) * evaluate_psf_full(psf, (j2,i2), L) * dict_ac[((j1 - shift1y, i1 - shift1x), (j2 + shift2y - shift1y, i2 + shift2x - shift1x))]
                                    T3k_extra = T3k_extra + evaluate_psf_full(psf, (j1,i1), L) * evaluate_psf_full(psf, (j2,i2), L) * dict_grads[((j1 - shift1y, i1 - shift1x), (j2 + shift2y - shift1y, i2 + shift2x - shift1x))]
                                    
        # %% 6.
        for j1 in range(shift2y - (L-1), L + shift2y):
            for i1 in range(shift2x - (L-1), L + shift2x):
                if not (np.abs(i1) < L and np.abs(j1) < L):
                    for j2 in range(max(j1, shift2y) - (L-1) - shift1y, L + min(j1, shift2y) - shift1y):
                        for i2 in range(max(i1, shift2x) - (L-1) - shift1x, L + min(i1, shift2x) - shift1x):
                            if not (np.abs(i2) < L and np.abs(j2) < L):
                                if not (np.abs(i2 - i1) < L and np.abs(j2 - j1) < L):
                                    M3k_extra = M3k_extra + evaluate_psf_full(psf, (j1,i1), L) * evaluate_psf_full(psf, (j2,i2), L) * dict_ac[((j1 - shift2y, i1 - shift2x), (j2 + shift1y - shift2y, i2 + shift1x - shift2x))]
                                    T3k_extra = T3k_extra + evaluate_psf_full(psf, (j1,i1), L) * evaluate_psf_full(psf, (j2,i2), L) * dict_grads[((j1 - shift2y, i1 - shift2x), (j2 + shift1y - shift2y, i2 + shift1x - shift2x))]
                                
    M3k = M3_clean + M3k_extra
    R3k = (gamma.T @ M3k) - M3data_k #+ bias3[k]
    T3k = (T3_clean + T3k_extra) * gamma.T # scale column k by gamma(k)
    
    return (M3k[0], R3k, T3k)

def cost_grad_fun_2d_full_psf(Z, params, L, K, list2_all=None, list3_all=None, list3_all_reversed=None, pool=None):
    # %% Definitions
    M1data = params[0]
    M2data = params[1]
    M3data = params[2]
    list2 = params[3]
    list3 = params[4]
    
    if list2_all == None:
        list2_all = calclist2_all(L)
    if list3_all == None:
        list3_all = calclist3_all(L)
        list3_all_reversed = [t[::-1] for t in list3_all]
    
    gamma = np.reshape(Z[:K], (K,))
    X = np.reshape(Z[K:K+L**2], (L,L,K)) ###### NOTICE for K > 1
    psf = np.reshape(Z[K+L**2:], (4*L-3, 4*L-3))
    if pool == None:
        dict_ac, dict_grads = calc_acs_grads(X, L, list2_all, list3_all)
    else:
        dict_ac, dict_grads = calc_acs_grads_parallel(X, L, pool, list2_all, list3_all, list3_all_reversed)
    n2 = np.shape(list2)[0]
    n3 = np.shape(list3)[0]
    w1 = 1/2 
    w2 = 1/(2*n2)
    w3 = 1/(2*n3)

    # %% First-order moment, forward model
    mean_X = np.sum(X, axis=(0, 1))/(L**2)
    M1 = gamma.T @ mean_X
    R1 = M1 - M1data
    
    # if pool == None:
    # %% Second-order moments, forward model
    M2s = [calc_2k_full_psf(k, list2[k], dict_ac, dict_grads, L, K, psf, M2data[k], gamma) for k in range(n2)]

    # %% Third-order moments, forward model
    M3s = [calc_3k_full_psf(k, list3[k][0], list3[k][1], dict_ac, dict_grads, L, K, psf, M3data[k], gamma) for k in range(n3)]
    # else:
    #     # %% Second-order moments, forward model
    #     M2s = pool.starmap(calc_2k, [(k, list2[k], dict_ac, dict_grads, L, K, psf, M2data[k], gamma) for k in range(n2)])
    
    #     # %% Third-order moments, forward model
    #     M3s = pool.starmap(calc_3k, [(k, list3[k][0], list3[k][1], dict_ac, dict_grads, L, K, psf, M3data[k], gamma, triplets) for k in range(n3)])
    
    # %% Calculations of cost and gradients
    M2 = np.asarray([M[0] for M in M2s])
    R2 = np.asarray([M[1] for M in M2s])
    T2 = [M[2] for M in M2s]
    g_psf_mat_2 = [M[3] for M in M2s]
    M3 = np.asarray([M[0] for M in M3s])
    R3 = np.asarray([M[1] for M in M3s])
    T3 = [M[2] for M in M3s]
    g_psf_mat_3 = [M[3] for M in M3s]
    
    f = w1*R1**2 + w2*R2@R2 + w3*R3@R3
    
    R2T2 = 0
    R2g_2 = 0
    for k in range(n2):
        R2T2 = R2T2 + R2[k]*T2[k]
        R2g_2 = R2g_2 + R2[k]*g_psf_mat_2[k]
    
    R3T3 = 0
    R3g_3 = 0
    for k in range(n3):
        R3T3 = R3T3 + R3[k]*T3[k]
        R3g_3 = R3g_3 + R3[k]*g_psf_mat_3[k]
    
    g_X = (2*w1*R1/(L**2)) * np.ones((L, L, K)) * gamma.T \
        + 2*w2*R2T2 \
            + 2*w3*R3T3
    g_gamma = 2*w1*R1*mean_X \
        + 2*w2*R2@M2 \
            + 2*w3*R3@M3
    g_psf = \
        + 2*w2*R2g_2 \
            + 2*w3*R3g_3

    return f, np.concatenate((np.reshape(g_gamma, (K,)), g_X.flatten(), g_psf.flatten()))

def calc_2k_full_psf(k, shift1, dict_ac, dict_grads, L, K, psf, M2data_k, gamma):
    shift1y = shift1[0]
    shift1x = shift1[1]
    
    # %% 1.
    M2_clean = dict_ac[shift1]
    T2_clean = dict_grads[shift1]
    
    M2k_extra = np.zeros(K)
    T2k_extra = np.zeros((L, L, K))
    g_psf_mat = np.zeros_like(psf)
    
    # %% 2.
    for j in range(shift1y-(L-1), L + shift1y):
        for i in range(shift1x-(L-1), L + shift1x):
            if not (np.abs(i) < L and np.abs(j) < L):
                M2k_extra = M2k_extra + evaluate_psf_full(psf, (j,i), L) \
                * dict_ac[(j-shift1y, i-shift1x)]#evaluate_psf(psf, np.sqrt(i**2 + j**2))
                T2k_extra = T2k_extra + evaluate_psf_full(psf, (j,i), L) \
                * dict_grads[(j-shift1y, i-shift1x)]
                g_psf_mat[j + 2*L-2, i + 2*L-2] = g_psf_mat[j + 2*L-2, i + 2*L-2] + dict_ac[(j-shift1y, i-shift1x)]
                
    M2k = M2_clean + M2k_extra
    R2k = (gamma.T @ M2k) - M2data_k #+ bias2[k]
    T2k = (T2_clean + T2k_extra) * gamma.T # scale column k by gamma(k)
    g_psf_mat2k = g_psf_mat*gamma.T
    
    return (M2k[0], R2k, T2k, g_psf_mat2k)

def calc_3k_full_psf(k, shift1, shift2, dict_ac, dict_grads, L, K, psf, M3data_k, gamma):
    shift1y = shift1[0]
    shift1x = shift1[1]

    shift2y = shift2[0]
    shift2x = shift2[1]
            
    # %% 1.
    M3_clean = dict_ac[(shift1, shift2)]
    T3_clean = dict_grads[(shift1, shift2)]
    
    # %% 2. ->
    M3k_extra = np.zeros(K)
    T3k_extra = np.zeros((L, L, K))
    g_psf_mat = np.zeros_like(psf)
    
    # %% 2. 
    for j in range(max(shift1y, shift2y) - (L-1), L + min(shift1y, shift2y)):
        for i in range(max(shift1x, shift2x) - (L-1), L + min(shift1x, shift2x)):
            if not (np.abs(i) < L and np.abs(j) < L):
                M3k_extra = M3k_extra + evaluate_psf_full(psf, (j,i), L) * dict_ac[((shift2y-shift1y, shift2x-shift1x), (j-shift1y, i-shift1x))]
                T3k_extra = T3k_extra + evaluate_psf_full(psf, (j,i), L) * dict_grads[((shift2y-shift1y, shift2x-shift1x), (j-shift1y, i-shift1x))]
                g_psf_mat[j + 2*L-2, i + 2*L-2] = g_psf_mat[j + 2*L-2, i + 2*L-2] + dict_ac[((shift2y-shift1y, shift2x-shift1x), (j-shift1y, i-shift1x))]
           
    # %% 3. 
    for j in range(shift1y - (L-1), L + shift1y - shift2y):
        for i in range(shift1x - (L-1), L + shift1x - shift2x):
            if not (np.abs(i) < L and np.abs(j) < L):
                M3k_extra = M3k_extra + evaluate_psf_full(psf, (j,i), L) * dict_ac[((j-shift1y, i-shift1x), (j+shift2y-shift1y, i+shift2x-shift1x))]
                T3k_extra = T3k_extra + evaluate_psf_full(psf, (j,i), L) * dict_grads[((j-shift1y, i-shift1x), (j+shift2y-shift1y, i+shift2x-shift1x))]
                g_psf_mat[j + 2*L-2, i + 2*L-2] = g_psf_mat[j + 2*L-2, i + 2*L-2] + dict_ac[((j-shift1y, i-shift1x), (j+shift2y-shift1y, i+shift2x-shift1x))]
    
    # %% 4. 
    for j in range(shift2y - (L-1), L + shift2y - shift1y):
        for i in range(shift2x - (L-1), L + shift2x - shift1x):
            if not (np.abs(i) < L and np.abs(j) < L):
                M3k_extra = M3k_extra + evaluate_psf_full(psf, (j,i), L) * dict_ac[((j-shift2y, i-shift2x), (j+shift1y-shift2y, i+shift1x-shift2x))]
                T3k_extra = T3k_extra + evaluate_psf_full(psf, (j,i), L) * dict_grads[((j-shift2y, i-shift2x), (j+shift1y-shift2y, i+shift1x-shift2x))]
                g_psf_mat[j + 2*L-2, i + 2*L-2] = g_psf_mat[j + 2*L-2, i + 2*L-2] + dict_ac[((j-shift2y, i-shift2x), (j+shift1y-shift2y, i+shift1x-shift2x))]
                  
    M3k = M3_clean + M3k_extra
    R3k = (gamma.T @ M3k) - M3data_k #+ bias3[k]
    T3k = (T3_clean + T3k_extra) * gamma.T # scale column k by gamma(k)
    g_psf_mat3k = g_psf_mat*gamma.T
    
    return (M3k[0], R3k, T3k, g_psf_mat3k)

def cost_grad_fun_2d_psf_expansion(Z, params, L, K, list2_all=None, list3_all=None, list3_all_reversed=None, pool=None):
    # %% Definitions
    M1data = params[0]
    M2data = params[1]
    M3data = params[2]
    list2 = params[3]
    list3 = params[4]
    
    if list2_all == None:
        list2_all = calclist2_all(L)
    if list3_all == None:
        list3_all = calclist3_all(L)
        list3_all_reversed = [t[::-1] for t in list3_all]
    
    gamma = np.reshape(Z[:K], (K,))
    X = np.reshape(Z[K:K+L**2], (L,L,K)) ###### NOTICE for K > 1
    coeffs = Z[K+L**2:]
    if pool == None:
        dict_ac, dict_grads = calc_acs_grads(X, L, list2_all, list3_all)
    else:
        dict_ac, dict_grads = calc_acs_grads_parallel(X, L, pool, list2_all, list3_all, list3_all_reversed)
    b = np.sqrt(2)*(2*L-1)
    n2 = np.shape(list2)[0]
    n3 = np.shape(list3)[0]
    w1 = 1/2 
    w2 = 1/(2*n2)
    w3 = 1/(2*n3)
    
    N = len(coeffs)
    roots = special.jn_zeros(0, N)

    # %% First-order moment, forward model
    mean_X = np.sum(X, axis=(0, 1))/(L**2)
    M1 = gamma.T @ mean_X
    R1 = M1 - M1data
    
    # if pool == None:
    # %% Second-order moments, forward model
    M2s = [calc_2k_psf_expansion(k, list2[k], dict_ac, dict_grads, L, K, coeffs, b, roots, N, M2data[k], gamma) for k in range(n2)]

    # %% Third-order moments, forward model
    M3s = [calc_3k_psf_expansion(k, list3[k][0], list3[k][1], dict_ac, dict_grads, L, K, coeffs, b, roots, N, M3data[k], gamma) for k in range(n3)]
    # else:
    #     # %% Second-order moments, forward model
    #     M2s = pool.starmap(calc_2k, [(k, list2[k], dict_ac, dict_grads, L, K, psf, M2data[k], gamma) for k in range(n2)])
    
    #     # %% Third-order moments, forward model
    #     M3s = pool.starmap(calc_3k, [(k, list3[k][0], list3[k][1], dict_ac, dict_grads, L, K, psf, M3data[k], gamma, triplets) for k in range(n3)])
    
    # %% Calculations of cost and gradients
    M2 = np.asarray([M[0] for M in M2s])
    R2 = np.asarray([M[1] for M in M2s])
    T2 = [M[2] for M in M2s]
    g_coeffs_part_2 = [M[3] for M in M2s]
    M3 = np.asarray([M[0] for M in M3s])
    R3 = np.asarray([M[1] for M in M3s])
    T3 = [M[2] for M in M3s]
    g_coeffs_part_3 = [M[3] for M in M3s]
    
    f = w1*R1**2 + w2*R2@R2 + w3*R3@R3
    
    R2T2 = 0
    R2g_2 = 0
    for k in range(n2):
        R2T2 = R2T2 + R2[k]*T2[k]
        R2g_2 = R2g_2 + R2[k]*g_coeffs_part_2[k]
    
    R3T3 = 0
    R3g_3 = 0
    for k in range(n3):
        R3T3 = R3T3 + R3[k]*T3[k]
        R3g_3 = R3g_3 + R3[k]*g_coeffs_part_3[k]
    
    g_X = (2*w1*R1/(L**2)) * np.ones((L, L, K)) * gamma.T \
        + 2*w2*R2T2 \
            + 2*w3*R3T3
    g_gamma = 2*w1*R1*mean_X \
        + 2*w2*R2@M2 \
            + 2*w3*R3@M3
    g_coeffs = \
        + 2*w2*R2g_2 \
            + 2*w3*R3g_3

    return f, np.concatenate((np.reshape(g_gamma, (K,)), g_X.flatten(), g_coeffs.flatten()))

def calc_2k_psf_expansion(k, shift1, dict_ac, dict_grads, L, K, coeffs, b, roots, N, M2data_k, gamma):
    shift1y = shift1[0]
    shift1x = shift1[1]
    
    # %% 1.
    M2_clean = dict_ac[shift1]
    T2_clean = dict_grads[shift1]
    
    M2k_extra = np.zeros(K)
    T2k_extra = np.zeros((L, L, K))
    g_coeffs_part = np.zeros_like(coeffs)
    
    # %% 2.
    for j in range(shift1y-(L-1), L + shift1y):
        for i in range(shift1x-(L-1), L + shift1x):
            if not (np.abs(i) < L and np.abs(j) < L):
                psf_i_j = fourier_bessel_expansion(coeffs, b, np.sqrt(i**2 + j**2))
                Js = np.array([special.j0(roots[n]*np.sqrt(i**2 + j**2)/b) for n in range(N)])
                g_coeffs_part = g_coeffs_part + dict_ac[(j-shift1y, i-shift1x)] * Js
                M2k_extra = M2k_extra + psf_i_j \
                * dict_ac[(j-shift1y, i-shift1x)]#evaluate_psf(psf, np.sqrt(i**2 + j**2))
                T2k_extra = T2k_extra + psf_i_j \
                * dict_grads[(j-shift1y, i-shift1x)]
                
    M2k = M2_clean + M2k_extra
    R2k = (gamma.T @ M2k) - M2data_k #+ bias2[k]
    T2k = (T2_clean + T2k_extra) * gamma.T # scale column k by gamma(k)
    g_coeffs_part_2k = g_coeffs_part*gamma.T
    
    return (M2k[0], R2k, T2k, g_coeffs_part_2k)

def calc_3k_psf_expansion(k, shift1, shift2, dict_ac, dict_grads, L, K, coeffs, b, roots, N, M3data_k, gamma):
    shift1y = shift1[0]
    shift1x = shift1[1]

    shift2y = shift2[0]
    shift2x = shift2[1]
            
    # %% 1.
    M3_clean = dict_ac[(shift1, shift2)]
    T3_clean = dict_grads[(shift1, shift2)]
    
    # %% 2. ->
    M3k_extra = np.zeros(K)
    T3k_extra = np.zeros((L, L, K))
    g_coeffs_part = np.zeros_like(coeffs)
    
    # %% 2. 
    for j in range(max(shift1y, shift2y) - (L-1), L + min(shift1y, shift2y)):
        for i in range(max(shift1x, shift2x) - (L-1), L + min(shift1x, shift2x)):
            if not (np.abs(i) < L and np.abs(j) < L):
                psf_i_j = fourier_bessel_expansion(coeffs, b, np.sqrt(i**2 + j**2))
                Js = np.array([special.j0(roots[n]*np.sqrt(i**2 + j**2)/b) for n in range(N)])
                g_coeffs_part = g_coeffs_part + dict_ac[((shift2y-shift1y, shift2x-shift1x), (j-shift1y, i-shift1x))] * Js
                M3k_extra = M3k_extra + psf_i_j * dict_ac[((shift2y-shift1y, shift2x-shift1x), (j-shift1y, i-shift1x))]
                T3k_extra = T3k_extra + psf_i_j * dict_grads[((shift2y-shift1y, shift2x-shift1x), (j-shift1y, i-shift1x))]
           
    # %% 3. 
    for j in range(shift1y - (L-1), L + shift1y - shift2y):
        for i in range(shift1x - (L-1), L + shift1x - shift2x):
            if not (np.abs(i) < L and np.abs(j) < L):
                psf_i_j = fourier_bessel_expansion(coeffs, b, np.sqrt(i**2 + j**2))
                Js = np.array([special.j0(roots[n]*np.sqrt(i**2 + j**2)/b) for n in range(N)])
                g_coeffs_part = g_coeffs_part + dict_ac[((j-shift1y, i-shift1x), (j+shift2y-shift1y, i+shift2x-shift1x))] * Js
                M3k_extra = M3k_extra + psf_i_j * dict_ac[((j-shift1y, i-shift1x), (j+shift2y-shift1y, i+shift2x-shift1x))]
                T3k_extra = T3k_extra + psf_i_j * dict_grads[((j-shift1y, i-shift1x), (j+shift2y-shift1y, i+shift2x-shift1x))]
    
    # %% 4. 
    for j in range(shift2y - (L-1), L + shift2y - shift1y):
        for i in range(shift2x - (L-1), L + shift2x - shift1x):
            if not (np.abs(i) < L and np.abs(j) < L):
                psf_i_j = fourier_bessel_expansion(coeffs, b, np.sqrt(i**2 + j**2))
                Js = np.array([special.j0(roots[n]*np.sqrt(i**2 + j**2)/b) for n in range(N)])
                g_coeffs_part = g_coeffs_part + dict_ac[((j-shift2y, i-shift2x), (j+shift1y-shift2y, i+shift1x-shift2x))] * Js
                M3k_extra = M3k_extra + psf_i_j * dict_ac[((j-shift2y, i-shift2x), (j+shift1y-shift2y, i+shift1x-shift2x))]
                T3k_extra = T3k_extra + psf_i_j * dict_grads[((j-shift2y, i-shift2x), (j+shift1y-shift2y, i+shift1x-shift2x))]
                  
    M3k = M3_clean + M3k_extra
    R3k = (gamma.T @ M3k) - M3data_k #+ bias3[k]
    T3k = (T3_clean + T3k_extra) * gamma.T # scale column k by gamma(k)
    g_coeffs_part_3k = g_coeffs_part*gamma.T
    
    return (M3k[0], R3k, T3k, g_coeffs_part_3k)



def cost_fun_2d_known_psf(Z, params, psf, triplets, L, K, list2_all=None, list3_all=None, list3_all_reversed=None, pool=None):
    # %% Definitions
    M1data = params[0]
    M2data = params[1]
    M3data = params[2]
    list2 = params[3]
    list3 = params[4]
    
    if list2_all == None:
        list2_all = calclist2_all(L)
    if list3_all == None:
        list3_all = calclist3_all(L)
        list3_all_reversed = [t[::-1] for t in list3_all]
    
    gamma = np.reshape(Z[:K], (K,))
    X = np.reshape(Z[K:], (L,L,K)) ###### NOTICE for K > 1
    if pool == None:
        dict_ac, dict_grads = calc_acs_grads(X, L, list2_all, list3_all)
    else:
        dict_ac, dict_grads = calc_acs_grads_parallel(X, L, pool, list2_all, list3_all, list3_all_reversed)
    n2 = np.shape(list2)[0]
    n3 = np.shape(list3)[0]
    w1 = 1/2 
    w2 = 1/(2*n2)
    w3 = 1/(2*n3)

    # %% First-order moment, forward model
    mean_X = np.sum(X, axis=(0, 1))/(L**2)
    M1 = gamma.T @ mean_X
    R1 = M1 - M1data
    
    # if pool == None:
    # %% Second-order moments, forward model
    M2s = [calc_2k_known_psf_Adam(k, list2[k], dict_ac, dict_grads, L, K, psf, M2data[k], gamma) for k in range(n2)]

    # %% Third-order moments, forward model
    M3s = [calc_3k_known_psf_Adam(k, list3[k][0], list3[k][1], dict_ac, dict_grads, L, K, psf, M3data[k], gamma, triplets) for k in range(n3)]
    # else:
    #     # %% Second-order moments, forward model
    #     M2s = pool.starmap(calc_2k, [(k, list2[k], dict_ac, dict_grads, L, K, psf, M2data[k], gamma) for k in range(n2)])
    
    #     # %% Third-order moments, forward model
    #     M3s = pool.starmap(calc_3k, [(k, list3[k][0], list3[k][1], dict_ac, dict_grads, L, K, psf, M3data[k], gamma, triplets) for k in range(n3)])
    
    # %% Calculations of cost and gradients
    R2 = np.asarray([M[1] for M in M2s])
    R3 = np.asarray([M[1] for M in M3s])
    
    f = w1*R1**2 + w2*R2@R2 + w3*R3@R3
    
    return f

def calc_2k_known_psf_Adam(k, shift1, dict_ac, dict_grads, L, K, psf, M2data_k, gamma):
    shift1y = shift1[0]
    shift1x = shift1[1]
    
    # %% 1.
    M2_clean = dict_ac[shift1]
    T2_clean = dict_grads[shift1]
    
    M2k_extra = np.zeros(K)
    T2k_extra = np.zeros((L, L, K))
    
    # %% 2.
    for j in range(shift1y-(L-1), L + shift1y):
        for i in range(shift1x-(L-1), L + shift1x):
            if not (np.abs(i) < L and np.abs(j) < L):
                M2k_extra = M2k_extra + evaluate_psf_full(psf, (j,i), L) \
                * dict_ac[(j-shift1y, i-shift1x)]#evaluate_psf(psf, np.sqrt(i**2 + j**2))
                T2k_extra = T2k_extra + evaluate_psf_full(psf, (j,i), L) \
                * dict_grads[(j-shift1y, i-shift1x)]
                
    M2k = M2_clean + M2k_extra
    R2k = (gamma.T @ M2k) - M2data_k #+ bias2[k]
    T2k = (T2_clean + T2k_extra) * gamma.T # scale column k by gamma(k)
    
    return (M2k[0], R2k)

def calc_3k_known_psf_Adam(k, shift1, shift2, dict_ac, dict_grads, L, K, psf, M3data_k, gamma, triplets):
    shift1y = shift1[0]
    shift1x = shift1[1]

    shift2y = shift2[0]
    shift2x = shift2[1]
            
    # %% 1.
    M3_clean = dict_ac[(shift1, shift2)]
        
    # %% 2. ->
    M3k_extra = np.zeros(K)
    
    # %% 2. 
    for j in range(max(shift1y, shift2y) - (L-1), L + min(shift1y, shift2y)):
        for i in range(max(shift1x, shift2x) - (L-1), L + min(shift1x, shift2x)):
            if not (np.abs(i) < L and np.abs(j) < L):
                M3k_extra = M3k_extra + evaluate_psf_full(psf, (j,i), L) * dict_ac[((shift2y-shift1y, shift2x-shift1x), (j-shift1y, i-shift1x))]
           
    # %% 3. 
    for j in range(shift1y - (L-1), L + shift1y - shift2y):
        for i in range(shift1x - (L-1), L + shift1x - shift2x):
            if not (np.abs(i) < L and np.abs(j) < L):
                M3k_extra = M3k_extra + evaluate_psf_full(psf, (j,i), L) * dict_ac[((j-shift1y, i-shift1x), (j+shift2y-shift1y, i+shift2x-shift1x))]
    
    # %% 4. 
    for j in range(shift2y - (L-1), L + shift2y - shift1y):
        for i in range(shift2x - (L-1), L + shift2x - shift1x):
            if not (np.abs(i) < L and np.abs(j) < L):
                M3k_extra = M3k_extra + evaluate_psf_full(psf, (j,i), L) * dict_ac[((j-shift2y, i-shift2x), (j+shift1y-shift2y, i+shift1x-shift2x))]
             
    if triplets:
        # %% 5.                 
        for j1 in range(shift1y - (L-1), L + shift1y):
            for i1 in range(shift1x - (L-1), L + shift1x):
                if not (np.abs(i1) < L and np.abs(j1) < L):
                    for j2 in range(max(j1, shift1y) - (L-1) - shift2y, L + min(j1, shift1y) - shift2y):
                        for i2 in range(max(i1, shift1x) - (L-1) - shift2x, L + min(i1, shift1x) - shift2x):
                            if not (np.abs(i2) < L and np.abs(j2) < L):
                                if not (np.abs(i2 - i1) < L and np.abs(j2 - j1) < L):
                                    M3k_extra = M3k_extra + evaluate_psf_full(psf, (j1,i1), L) * evaluate_psf_full(psf, (j2,i2), L) * dict_ac[((j1 - shift1y, i1 - shift1x), (j2 + shift2y - shift1y, i2 + shift2x - shift1x))]
                                    
        # $$ 6.
        for j1 in range(shift2y - (L-1), L + shift2y):
            for i1 in range(shift2x - (L-1), L + shift2x):
                if not (np.abs(i1) < L and np.abs(j1) < L):
                    for j2 in range(max(j1, shift2y) - (L-1) - shift1y, L + min(j1, shift2y) - shift1y):
                        for i2 in range(max(i1, shift2x) - (L-1) - shift1x, L + min(i1, shift2x) - shift1x):
                            if not (np.abs(i2) < L and np.abs(j2) < L):
                                if not (np.abs(i2 - i1) < L and np.abs(j2 - j1) < L):
                                    M3k_extra = M3k_extra + evaluate_psf_full(psf, (j1,i1), L) * evaluate_psf_full(psf, (j2,i2), L) * dict_ac[((j1 - shift2y, i1 - shift2x), (j2 + shift1y - shift2y, i2 + shift1x - shift2x))]
                                
    M3k = M3_clean + M3k_extra
    R3k = (gamma.T @ M3k) - M3data_k #+ bias3[k]
        
    return (M3k[0], R3k)
