# -*- coding: utf-8 -*-
"""
Created on Fri Jul 31 20:23:28 2020

@author: kreym
"""

import numpy as np
from scipy.optimize import minimize, check_grad
import multiprocessing as mp
import c_g_funcs_rot
import psf_functions_2d as psf_funcs
from funcs_calc_moments_rot import calck1k2k3, calcmap3, calck1k2k3_binned, calck1, calcN_mat
from makeExtraMat import makeExtraMat
from maketsfMat import maketsfMat
from maketsfMat_parallel import maketsfMat_parallel
from matGradCoeffs_psf import matGradCoeffs_psf
import scipy.special as special
from generate_clean_micrograph_2d import generate_clean_micrograph_2d_one_neighbor_rots, generate_clean_micrograph_2d_rots

import psf_functions_2d
import tsf_functions_2d

import stochastic_optimizers

# import tensorflow as tf
# import torch

# def optimize_2d_known_psf_rot(initial_guesses, Bk, kvals, M1_y, M2_y, M3_y, sigma2, full_psf, L, K, gtol=1e-10):
#     # Optimization assuming known psf
#     k1_map = calck1(L)
#     k1k2k3_map = calck1k2k3(L)
#     N_mat = calcN_mat(L)
#     checkgrad = check_grad(func, grad, initial_guesses, Bk, kvals, M1_y, M2_y, M3_y, sigma2, full_psf, L, K, N_mat, k1_map, k1k2k3_map)
#     print(f'check {checkgrad}')
#     return minimize(fun=c_g_funcs_rot.cost_grad_fun_2d_known_psf_rot, x0=initial_guesses, method='BFGS', jac=True, options={'disp': True, 'maxiter': 100, 'gtol': gtol}, args = (Bk, kvals, M1_y, M2_y, M3_y, sigma2, full_psf, L, K, N_mat, k1_map, k1k2k3_map))

# def optimize_2d_known_psf_rot_k(initial_guesses, Bk, kvals, M1_y, M2_y, M3_y, sigma2, full_psf, L, K, gtol=1e-10):
#     # Optimization assuming known psf
#     k1_map = calck1(L)
#     k1k2k3_map = calck1k2k3(L)
#     N_mat = calcN_mat(L)
#     return minimize(fun=c_g_funcs_rot.cost_grad_fun_2d_known_psf_rot_k, x0=initial_guesses, method='BFGS', jac=True, options={'disp': True, 'maxiter': 100, 'gtol': gtol}, args = (Bk, kvals, M1_y, M2_y, M3_y, sigma2, full_psf, L, K, N_mat, k1_map, k1k2k3_map))



# def optimize_2d_known_psf_rot_parallel(initial_guesses, Bk, kvals, M1_y, M2_y, M3_y, sigma2, full_psf, L, K, gtol=1e-10):
#     # Optimization assuming known psf
#     k1_map = calck1(L)
#     k1k2k3_map = calck1k2k3(L)
#     N_mat = calcN_mat(L)
#     return minimize(fun=c_g_funcs_rot.cost_grad_fun_2d_known_psf_rot_parallel, x0=initial_guesses, method='BFGS', jac=True, options={'disp': True, 'maxiter': 50, 'gtol': gtol, 'return_all': True}, args = (Bk, kvals, M1_y, M2_y, M3_y, sigma2, full_psf, L, K, N_mat, k1_map, k1k2k3_map))

# def optimize_2d_rot_parallel(initial_guesses, nu, Bk, kvals, M1_y, M2_y, M3_y, sigma2, L, K, gtol=1e-10):
#     # Optimization assuming known psf
#     k1_map = calck1(L)
#     k1k2k3_map = calck1k2k3(L)
#     N_mat = calcN_mat(L)
#     return minimize(fun=c_g_funcs_rot.cost_grad_fun_2d_rot_parallel, x0=initial_guesses, method='BFGS', jac=True, options={'disp': True, 'maxiter': 150, 'gtol': gtol}, args = (nu, Bk, kvals, M1_y, M2_y, M3_y, sigma2, L, K, N_mat, k1_map, k1k2k3_map))


# def func(Z, Bk, kvals, M1data, M2data, M3data, sigma2, psf, L, K, N_mat, k1_map, k1k2k3_map):
#     return c_g_funcs_rot.cost_grad_fun_2d_known_psf_rot(Z, Bk, kvals, M1data, M2data, M3data, sigma2, psf, L, K, N_mat=None, k1_map=None, k1k2k3_map=None)[0]
# def grad(Z, Bk, kvals, M1data, M2data, M3data, sigma2, psf, L, K, N_mat, k1_map, k1k2k3_map):
#     return c_g_funcs_rot.cost_grad_fun_2d_known_psf_rot(Z, Bk, kvals, M1data, M2data, M3data, sigma2, psf, L, K, N_mat=None, k1_map=None, k1k2k3_map=None)[1:]
    
# def optimize_2d_known_psf_mat_rot(initial_guesses, gamma, Bk, kvals, M1_y, M2_y, M3_y, sigma2, full_psf, L, K, ExtraMat=None, gtol=1e-10):
#     # Optimization assuming known psf
#     k1_map = calck1(L)
#     k1k2k3_map = calck1k2k3(L)
#     N_mat = calcN_mat(L)
#     if ExtraMat == None:
#         ExtraMat = makeExtraMat(L, full_psf)

#     return minimize(fun=c_g_funcs_rot.cost_grad_fun_2d_known_psf_mat_rot, x0=initial_guesses, method='BFGS', jac=True, options={'disp': True, 'maxiter':1000, 'gtol': gtol}, args = (gamma, Bk, kvals, M1_y, M2_y, M3_y, sigma2, ExtraMat, full_psf, L, K, N_mat, k1_map, k1k2k3_map))

def optimize_2d_known_psf_mat_rot_gamma(initial_guesses, Bk, T, kvals, M1_y, M2_y, M3_y, sigma2, full_psf, L, K, ExtraMat2=None, ExtraMat3=None, numiters=3000, gtol=1e-15):
    # Optimization assuming known psf
    k1_map = calck1(L)
    k1k2k3_map = calck1k2k3(L)
    N_mat = calcN_mat(L)
    if ExtraMat2 == None or ExtraMat3 == None:
        ExtraMat2, ExtraMat3 = makeExtraMat(L, full_psf)

    return minimize(fun=c_g_funcs_rot.cost_grad_fun_2d_known_psf_mat_rot_gamma, x0=initial_guesses, method='BFGS', jac=True, options={'disp': True, 'maxiter':numiters, 'gtol': gtol}, args = (Bk, T, kvals, M1_y, M2_y, M3_y, sigma2, ExtraMat2, ExtraMat3, full_psf, L, K, N_mat, k1_map, k1k2k3_map))

def optimize_2d_known_psf_triplets(initial_guesses, Bk, T, kvals, M1_y, M2_y, M3_y, sigma2, L, K, tsfMat, ExtraMat2, ExtraMat3, numiters=3000, gtol=1e-15):
    # Optimization assuming known psf
    k1_map = calck1(L)
    map3 = calcmap3(L)
    N_mat = calcN_mat(L)
    
    return minimize(fun=c_g_funcs_rot.cost_grad_fun_2d_known_psf_tripletsnotparallelnew, x0=initial_guesses, method='BFGS', jac=True, options={'disp': True, 'maxiter':numiters, 'gtol': gtol}, args = (Bk, T, kvals, M1_y, M2_y, M3_y, sigma2, ExtraMat2, ExtraMat3, tsfMat, L, K, N_mat, k1_map, map3))

def optimize_2d_psf(initial_guesses, gamma, S1, S2_x, S2_x_neigh, S3_x, S3_x_neigh, Bk, kvals, M1data, M2data, M3data, sigma2, L, K, N_mat=None, numiters=100, gtol=1e-15):
    if N_mat == None:
        N_mat = calcN_mat(L)
    
    return minimize(fun=c_g_funcs_rot.cost_grad_fun_2d_psf, x0=initial_guesses, method='BFGS', jac=True, options={'disp': True, 'maxiter':numiters, 'gtol': gtol}, args = (gamma, S1, S2_x, S2_x_neigh, S3_x, S3_x_neigh, Bk, kvals, M1data, M2data, M3data, sigma2, L, K, N_mat))

def optimize_2d_x_psf(initial_guesses, Bk, T, kvals, M1_y, M2_y, M3_y, sigma2, psf_init, L, K, ExtraMat2=None, ExtraMat3=None, gtol=1e-15):
    k1_map = calck1(L)
    k1k2k3_map = calck1k2k3(L)
    N_mat = calcN_mat(L)
    if ExtraMat2 == None or ExtraMat3 == None:
        ExtraMat2, ExtraMat3 = makeExtraMat(L, psf_init)
        
    first_estimates = minimize(fun=c_g_funcs_rot.cost_grad_fun_2d_known_psf_mat_rot_gamma, x0=initial_guesses, method='BFGS', jac=True, options={'disp': True, 'maxiter':80, 'gtol': gtol}, args = (Bk, T, kvals, M1_y, M2_y, M3_y, sigma2, ExtraMat2, ExtraMat3, psf_init, L, K, N_mat, k1_map, k1k2k3_map))
    first_gamma = first_estimates.x[0]
    first_c = first_estimates.x[1: ]
    first_z = T.H@first_c 
    S1 = np.real(np.sum(np.fft.ifftn(Bk @ first_z), axis=(0, 1))/(L**2))    
    S2_x, _, S2_x_neigh, _, S3_x, _, S3_x_neigh, _ = c_g_funcs_rot.calc_acs_grads_rot_parallel(Bk, first_z, kvals, L, k1_map=k1_map, k1k2k3_map=k1k2k3_map)
    S3_x = np.real(S3_x)
    S3_x_neigh = np.real(S3_x_neigh)
    
    psf_estimates = optimize_2d_psf(psf_init.flatten(), first_gamma, S1, S2_x, S2_x_neigh, S3_x, S3_x_neigh, Bk, kvals, M1_y, M2_y, M3_y, sigma2, L, K, N_mat=N_mat, numiters=200, gtol=1e-15)
    
    psf_new = np.reshape(psf_estimates.x, (4*L-3, 4*L-3))
    ExtraMat2_new, ExtraMat3_new = makeExtraMat(L, psf_new)
    
    new_guesses = np.concatenate((np.reshape(first_gamma, (1,)), first_c))
    
    second_estimates = minimize(fun=c_g_funcs_rot.cost_grad_fun_2d_known_psf_mat_rot_gamma, x0=new_guesses, method='BFGS', jac=True, options={'disp': True, 'maxiter':2000, 'gtol': gtol}, args = (Bk, T, kvals, M1_y, M2_y, M3_y, sigma2, ExtraMat2_new, ExtraMat3_new, psf_new, L, K, N_mat, k1_map, k1k2k3_map))

    second_gamma = second_estimates.x[0]
    second_c = second_estimates.x[1: ]
    second_z = T.H@second_c 
    S1 = np.real(np.sum(np.fft.ifftn(Bk @ second_z), axis=(0, 1))/(L**2))    
    S2_x, _, S2_x_neigh, _, S3_x, _, S3_x_neigh, _ = c_g_funcs_rot.calc_acs_grads_rot_parallel(Bk, second_z, kvals, L, k1_map=k1_map, k1k2k3_map=k1k2k3_map)
    S3_x = np.real(S3_x)
    S3_x_neigh = np.real(S3_x_neigh)
    
    psf_estimates2 = optimize_2d_psf(psf_new.flatten(), second_gamma, S1, S2_x, S2_x_neigh, S3_x, S3_x_neigh, Bk, kvals, M1_y, M2_y, M3_y, sigma2, L, K, N_mat=N_mat, numiters=200, gtol=1e-15)
    
    psf_new2 = np.reshape(psf_estimates2.x, (4*L-3, 4*L-3))
    ExtraMat2_new, ExtraMat3_new = makeExtraMat(L, psf_new2)
    
    new_guesses = np.concatenate((np.reshape(second_gamma, (1,)), second_c))
    
    return minimize(fun=c_g_funcs_rot.cost_grad_fun_2d_known_psf_mat_rot_gamma, x0=new_guesses, method='BFGS', jac=True, options={'disp': True, 'maxiter':2000, 'gtol': gtol}, args = (Bk, T, kvals, M1_y, M2_y, M3_y, sigma2, ExtraMat2_new, ExtraMat3_new, psf_new, L, K, N_mat, k1_map, k1k2k3_map)), psf_estimates2


def optimize_2d_x_update_psf(initial_guesses, Bk, T, kvals, M1_y, M2_y, M3_y, sigma2, L, K, W, N, iters_till_change=150, gtol=1e-15):
    k1_map = calck1(L)
    k1k2k3_map = calck1k2k3(L)
    N_mat = calcN_mat(L)
    
    y_init, _, locs_init = generate_clean_micrograph_2d_one_neighbor_rots(initial_guesses[1: ], kvals, Bk, W, L, N, (initial_guesses[0]*(N/L)**2).astype(int), T)
    psf_init = psf_functions_2d.full_psf_2d(locs_init, L)
    ExtraMat2, ExtraMat3 = makeExtraMat(L, psf_init)
        
    first_estimates = minimize(fun=c_g_funcs_rot.cost_grad_fun_2d_known_psf_mat_rot_gamma, x0=initial_guesses, method='BFGS', jac=True, options={'disp': True, 'maxiter':iters_till_change, 'gtol': gtol}, args = (Bk, T, kvals, M1_y, M2_y, M3_y, sigma2, ExtraMat2, ExtraMat3, psf_init, L, K, N_mat, k1_map, k1k2k3_map))
    first_gamma = first_estimates.x[0]
    print(f'We got to gamma of {first_gamma}')
    first_c = first_estimates.x[1: ]
    
    y2, _, locs2 = generate_clean_micrograph_2d_one_neighbor_rots(first_c, kvals, Bk, W, L, N, (first_gamma*(N/L)**2).astype(int), T)
    psf2 = psf_functions_2d.full_psf_2d(locs2, L)
    ExtraMat2, ExtraMat3 = makeExtraMat(L, psf2)
    
    new_guesses = np.concatenate((np.reshape(first_gamma, (1,)), first_c))
    
    return minimize(fun=c_g_funcs_rot.cost_grad_fun_2d_known_psf_mat_rot_gamma, x0=new_guesses, method='BFGS', jac=True, options={'disp': True, 'maxiter':2000, 'gtol': gtol}, args = (Bk, T, kvals, M1_y, M2_y, M3_y, sigma2, ExtraMat2, ExtraMat3, psf2, L, K, N_mat, k1_map, k1k2k3_map)), psf2
    

def optimize_2d_x_update_psf_tsf(initial_guesses, Bk, T, kvals, M1_y, M2_y, M3_y, sigma2, L, K, W, N, iters_till_change=150, gtol=1e-15):
    k1_map = calck1(L)
    k1k2k3_map = calck1k2k3(L)
    N_mat = calcN_mat(L)
    
    y_init, _, locs_init = generate_clean_micrograph_2d_rots(initial_guesses[1: ], kvals, Bk, W, L, N, (initial_guesses[0]*(N/L)**2).astype(int), T)
    psf_init = psf_functions_2d.full_psf_2d(locs_init, L)
    ExtraMat2, ExtraMat3 = makeExtraMat(L, psf_init)
    tsf_init = tsf_functions_2d.full_tsf_2d(locs_init, L)
    tsfMat = maketsfMat(L, tsf_init)
        
    first_estimates = minimize(fun=c_g_funcs_rot.cost_grad_fun_2d_known_psf_triplets, x0=initial_guesses, method='BFGS', jac=True, options={'disp': True, 'maxiter':iters_till_change, 'gtol': gtol}, args = (Bk, T, kvals, M1_y, M2_y, M3_y, sigma2, ExtraMat2, ExtraMat3, tsfMat, L, K, N_mat, k1_map, k1k2k3_map))
    first_gamma = first_estimates.x[0]
    print(f'We got to gamma of {first_gamma}')
    first_c = first_estimates.x[1: ]
    
    y2, _, locs2 = generate_clean_micrograph_2d_rots(first_c, kvals, Bk, W, L, N, (first_gamma*(N/L)**2).astype(int), T)
    psf2 = psf_functions_2d.full_psf_2d(locs2, L)
    ExtraMat2, ExtraMat3 = makeExtraMat(L, psf2)
    tsf2 = tsf_functions_2d.full_tsf_2d(locs2, L)
    tsfMat = maketsfMat(L, tsf2)
    
    new_guesses = np.concatenate((np.reshape(first_gamma, (1,)), first_c))
    
    return minimize(fun=c_g_funcs_rot.cost_grad_fun_2d_known_psf_triplets, x0=new_guesses, method='BFGS', jac=True, options={'disp': True, 'maxiter':2000, 'gtol': gtol}, args = (Bk, T, kvals, M1_y, M2_y, M3_y, sigma2, ExtraMat2, ExtraMat3, tsfMat, L, K, N_mat, k1_map, k1k2k3_map)), psf2, tsf2
    
def optimize_2d_x_update_psf_tsfnew(initial_guesses, Bk, T, kvals, M1_y, M2_y, M3_y, sigma2, L, K, W, N, iters_till_change=150, gtol=1e-15):
    k1_map = calck1(L)
    map3 = calcmap3(L)
    N_mat = calcN_mat(L)
    
    y_init, _, locs_init = generate_clean_micrograph_2d_rots(initial_guesses[1: ], kvals, Bk, W, L, 10000, (initial_guesses[0]*(10000/L)**2).astype(int), T)
    psf_init = psf_functions_2d.full_psf_2d(locs_init, L)
    ExtraMat2, ExtraMat3 = makeExtraMat(L, psf_init)
    tsf_init = tsf_functions_2d.full_tsf_2d(locs_init, L)
    tsfMat = maketsfMat_parallel(L, tsf_init)
        
    first_estimates = minimize(fun=c_g_funcs_rot.cost_grad_fun_2d_known_psf_tripletsnew, x0=initial_guesses, method='BFGS', jac=True, options={'disp': True, 'maxiter':iters_till_change, 'gtol': gtol}, args = (Bk, T, kvals, M1_y, M2_y, M3_y, sigma2, ExtraMat2, ExtraMat3, tsfMat, L, K, N_mat, k1_map, map3))
    first_gamma = first_estimates.x[0]
    print(f'We got to gamma of {first_gamma}')
    first_c = first_estimates.x[1: ]
    
    y2, _, locs2 = generate_clean_micrograph_2d_rots(first_c, kvals, Bk, W, L, N, (first_gamma*(N/L)**2).astype(int), T)
    psf2 = psf_functions_2d.full_psf_2d(locs2, L)
    ExtraMat2, ExtraMat3 = makeExtraMat(L, psf2)
    tsf2 = tsf_functions_2d.full_tsf_2d(locs2, L)
    tsfMat = maketsfMat_parallel(L, tsf2)
    
    new_guesses = np.concatenate((np.reshape(first_gamma, (1,)), first_c))
    
    return minimize(fun=c_g_funcs_rot.cost_grad_fun_2d_known_psf_tripletsnew, x0=new_guesses, method='BFGS', jac=True, options={'disp': True, 'maxiter':2000, 'gtol': gtol}, args = (Bk, T, kvals, M1_y, M2_y, M3_y, sigma2, ExtraMat2, ExtraMat3, tsfMat, L, K, N_mat, k1_map, map3)), psf2, tsf2
   
def optimize_2d_x_update_psf_tsfnotparallelnew(initial_guesses, Bk, T, kvals, M1_y, M2_y, M3_y, sigma2, L, K, W, N, iters_till_change=150, gtol=1e-15):
    k1_map = calck1(L)
    map3 = calcmap3(L)
    N_mat = calcN_mat(L)
    
    print('Starting...')
    
    y_init, _, locs_init = generate_clean_micrograph_2d_rots(initial_guesses[1: ], kvals, Bk, W, L, 10000, (initial_guesses[0]*(10000/L)**2).astype(int), T)
    psf_init = psf_functions_2d.full_psf_2d(locs_init, L)
    ExtraMat2, ExtraMat3 = makeExtraMat(L, psf_init)
    tsf_init = tsf_functions_2d.full_tsf_2d(locs_init, L)
    tsfMat = maketsfMat(L, tsf_init)
    print('Done')
        
    first_estimates = minimize(fun=c_g_funcs_rot.cost_grad_fun_2d_known_psf_tripletsnotparallelnew, x0=initial_guesses, method='BFGS', jac=True, options={'disp': True, 'maxiter':iters_till_change, 'gtol': gtol}, args = (Bk, T, kvals, M1_y, M2_y, M3_y, sigma2, ExtraMat2, ExtraMat3, tsfMat, L, K, N_mat, k1_map, map3))
    first_gamma = first_estimates.x[0]
    print(f'We got to gamma of {first_gamma}')
    first_c = first_estimates.x[1: ]
    
    y2, _, locs2 = generate_clean_micrograph_2d_rots(first_c, kvals, Bk, W, L, 10000, (first_gamma*(10000/L)**2).astype(int), T)
    psf2 = psf_functions_2d.full_psf_2d(locs2, L)
    ExtraMat2, ExtraMat3 = makeExtraMat(L, psf2)
    tsf2 = tsf_functions_2d.full_tsf_2d(locs2, L)
    tsfMat = maketsfMat(L, tsf2)
    
    new_guesses = np.concatenate((np.reshape(first_gamma, (1,)), first_c))
    
    return minimize(fun=c_g_funcs_rot.cost_grad_fun_2d_known_psf_tripletsnotparallelnew, x0=new_guesses, method='BFGS', jac=True, options={'disp': True, 'maxiter':2000, 'gtol': gtol}, args = (Bk, T, kvals, M1_y, M2_y, M3_y, sigma2, ExtraMat2, ExtraMat3, tsfMat, L, K, N_mat, k1_map, map3)), psf2, tsf2
   
def optimize_2d_x_known_psf_binnednotparallel(initial_guesses, Bk, T, kvals, M1_y, M2_y, M3_y, sigma2, L, R, K, W, N, ExtraMat2, ExtraMat3, gtol=1e-15):
    k1_map = calck1(L)
    k1k2k3_map_binned, k1k2k3_map_binned_equi_idxs = calck1k2k3_binned(L, R)
    N_mat = calcN_mat(L)
    
    return minimize(fun=c_g_funcs_rot.cost_grad_fun_2d_known_psf_binnednotparallel, x0=initial_guesses, method='BFGS', jac=True, options={'disp': True, 'maxiter':2000, 'gtol': gtol}, args = (Bk, T, kvals, M1_y, M2_y, M3_y, sigma2, ExtraMat2, ExtraMat3, L, K, N_mat, k1_map, k1k2k3_map_binned, k1k2k3_map_binned_equi_idxs))
   
def optimize_2d_known_psf_triplets_with_callback(initial_guesses, Bk, T, kvals, M1_y, M2_y, M3_y, sigma2, L, K, tsfMat, ExtraMat2, ExtraMat3, numiters=3000, gtol=1e-15):
    # Optimization assuming known psf
    k1_map = calck1(L)
    map3 = calcmap3(L)
    N_mat = calcN_mat(L)
    history = [initial_guesses[0]]
    def func_callback(x):
        history.append(x[0])
    return minimize(fun=c_g_funcs_rot.cost_grad_fun_2d_known_psf_tripletsnotparallelnew, x0=initial_guesses, method='BFGS', jac=True, callback=func_callback, options={'disp': True, 'maxiter':numiters, 'gtol': gtol}, args = (Bk, T, kvals, M1_y, M2_y, M3_y, sigma2, ExtraMat2, ExtraMat3, tsfMat, L, K, N_mat, k1_map, map3)), history


# def optimize_2d_known_psf_mat_rot_gamma_parallel(initial_guesses, Bk, kvals, M1_y, M2_y, M3_y, sigma2, full_psf, L, K, ExtraMat=None, gtol=1e-10):
#     # Optimization assuming known psf
#     k1_map = calck1(L)
#     k1k2k3_map = calck1k2k3(L)
#     N_mat = calcN_mat(L)
#     if ExtraMat == None:
#         ExtraMat = makeExtraMat(L, full_psf)

#     return minimize(fun=c_g_funcs_rot.cost_grad_fun_2d_known_psf_mat_rot_gamma_parallel, x0=initial_guesses, method='BFGS', jac=True, options={'disp': True, 'maxiter':500, 'gtol': gtol}, args = (Bk, kvals, M1_y, M2_y, M3_y, sigma2, ExtraMat, full_psf, L, K, N_mat, k1_map, k1k2k3_map))


# def optimize_2d_known_psf_mat_rot_parallel(initial_guesses, gamma, Bk, kvals, M1_y, M2_y, M3_y, sigma2, full_psf, L, K, ExtraMat=None, gtol=1e-10):
#     # Optimization assuming known psf
#     k1_map = calck1(L)
#     k1k2k3_map = calck1k2k3(L)
#     N_mat = calcN_mat(L)
#     if ExtraMat == None:
#         ExtraMat = makeExtraMat(L, full_psf)

#     return minimize(fun=c_g_funcs_rot.cost_grad_fun_2d_known_psf_mat_rot_parallel, x0=initial_guesses, method='BFGS', jac=True, options={'disp': True, 'maxiter':500, 'gtol': gtol}, args = (gamma, Bk, kvals, M1_y, M2_y, M3_y, sigma2, ExtraMat, full_psf, L, K, N_mat, k1_map, k1k2k3_map))

# def optimize_2d_known_psf_mat_rot_binned(initial_guesses, gamma, Bk, kvals, M1_y, M2_y, M3_y, sigma2, full_psf, L, K, R, gtol=1e-10):
#     # Optimization assuming known psf
#     k1_map = calck1(L)
#     k1k2k3_map_binned, k1k2k3_map_binned_equi_idxs = calck1k2k3_binned(L, R)
#     N_mat = calcN_mat(L)
#     ExtraMat = makeExtraMat(L, full_psf)

#     return minimize(fun=c_g_funcs_rot.cost_grad_fun_2d_known_psf_mat_rot_binned, x0=initial_guesses, method='BFGS', jac=True, options={'disp': True, 'gtol': gtol}, args = (gamma, Bk, kvals, M1_y, M2_y, M3_y, sigma2, ExtraMat, full_psf, L, K, N_mat, k1_map, k1k2k3_map_binned, k1k2k3_map_binned_equi_idxs))

# def optimize_2d_known_psf_mat_rot_stoch(initial_guesses, gamma, Bk, kvals, M1_y, M2_y, M3_y, sigma2, full_psf, L, K, gtol=1e-10):
#     # Optimization assuming known psf
#     k1_map = calck1(L)
#     k1k2k3_map = calck1k2k3(L)
#     N_mat = calcN_mat(L)
#     ExtraMat = makeExtraMat(L, full_psf)
    
#     return stochastic_optimizers.adam(fun=c_g_funcs_rot.cost_grad_fun_2d_known_psf_mat_rot, x0=initial_guesses, jac=True, args = (gamma, Bk, kvals, M1_y, M2_y, M3_y, sigma2, ExtraMat, full_psf, L, K, N_mat, k1_map, k1k2k3_map))

# def optimize_2d_rot_psf_coeffs(initial_guesses, gamma, S1, S3_x, S3_x_neigh, Bk, kvals, M1_y, M2_y, M3_y, sigma2, L, K, matCoeffs=None, gtol=1e-10):
#     # Optimization assuming known psf
#     k1_map = calck1(L)
#     k1k2k3_map = calck1k2k3(L)
#     N_mat = calcN_mat(L)
#     if matCoeffs == None:
#         N_coeffs = len(initial_guesses)
#         roots = special.jn_zeros(0, N_coeffs)
#         b = np.sqrt(2)*(2*L - 1)
#         matCoeffs = matGradCoeffs_psf(L, N_coeffs, roots, b, S3_x_neigh)

#     return minimize(fun=c_g_funcs_rot.cost_grad_fun_2d_rot_psf_coeffs, x0=initial_guesses, method='BFGS', jac=True, options={'disp': True, 'maxiter':500, 'gtol': gtol}, args = (gamma, S1, S3_x, S3_x_neigh, Bk, kvals, M1_y, M2_y, M3_y, sigma2, L, K, matCoeffs, N_mat, k1_map, k1k2k3_map))


























# def optimize_2d_known_psf_mat_rot_adam(initial_guesses, gamma, Bk, kvals, M1_y, M2_y, M3_y, sigma2, full_psf, L, K, gtol=1e-10):
#     # Optimization assuming known psf
#     k1_map = calck1(L)
#     k1k2k3_map = calck1k2k3(L)
#     N_mat = calcN_mat(L)
    
#     z_torch = z_torch = torch.from_numpy(initial_guesses)
#     z_torch.requires_grad_()
#     optimizer = torch.optim.SGD([z_torch], 1e-2)
#     for e in range(10):
#         loss = c_g_funcs_rot.cost_grad_fun_2d_known_psf_rot_adam(z_torch, torch.tensor(gamma), torch.tensor(Bk), torch.tensor(kvals), torch.tensor(M1_y), torch.tensor(M2_y), torch.tensor(M3_y), torch.tensor(sigma2), torch.tensor(full_psf), torch.tensor(L), torch.tensor(K), torch.tensor(N_mat), torch.tensor(k1_map), torch.tensor(k1k2k3_map))
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#     return optimizer.step(c_g_funcs_rot.cost_grad_fun_2d_known_psf_mat_rot_adam(z_torch, gamma, Bk, kvals, M1_y, M2_y, M3_y, sigma2, ExtraMat, full_psf, L, K, N_mat, k1_map, k1k2k3_map))

#     args = (gamma, Bk, kvals, M1_y, M2_y, M3_y, sigma2, ExtraMat, full_psf, L, K, N_mat, k1_map, k1k2k3_map))

# def optimize_2d_known_psf_parallel(initial_guesses, M1_Y, M2_Y, M3_Y, list2, list3, full_psf, triplets, L, K, gtol=1e-10):
#     # Optimization assuming known psf - using parallel processing
#     list2_all = calclist2_all(L)
#     list3_all = calclist3_all(L)
#     list3_all_reversed = [t[::-1] for t in list3_all]
#     pool = mp.Pool()
#     return minimize(fun=c_g_funcs.cost_grad_fun_2d_known_psf, x0=initial_guesses, method='BFGS', jac=True, options={'disp': False, 'gtol': gtol}, args = ([M1_Y, M2_Y, M3_Y, list2, list3, 0, 0], full_psf, triplets, L, K, list2_all, list3_all, list3_all_reversed, pool))

# def optimize_2d_full_psf_estimation(initial_guesses, M1_Y, M2_Y, M3_Y, list2, list3, L, K, psf0=None, gtol=1e-10):
#     # Optimization + estimating full (2-D) psf
#     if psf0 is None:
#         psf0, _ = psf_funcs.psf_initial_guess(L, initial_guesses[0])
#     initial_guesses = np.concatenate((initial_guesses, psf0.flatten()))
#     list2_all = calclist2_all(L)
#     list3_all = calclist3_all(L)
#     # list3_all_reversed = [t[::-1] for t in list3_all]
#     return minimize(fun=c_g_funcs.cost_grad_fun_2d_full_psf, x0=initial_guesses, method='BFGS', jac=True, options={'disp': False, 'gtol': gtol}, args = ([M1_Y, M2_Y, M3_Y, list2, list3, 0, 0], L, K, list2_all, list3_all))

# def optimize_2d_full_psf_no_estimation(initial_guesses, M1_Y, M2_Y, M3_Y, list2, list3, triplets, L, K, full_psf=None, gtol=1e-10):
#     # Optimization using some initial psf, without estimating psf
#     if full_psf is None:
#         full_psf, _ = psf_funcs.psf_initial_guess(L, initial_guesses[0])
#     list2_all = calclist2_all(L)
#     list3_all = calclist3_all(L)
#     # list3_all_reversed = [t[::-1] for t in list3_all]
#     return minimize(fun=c_g_funcs.cost_grad_fun_2d_known_psf, x0=initial_guesses, method='BFGS', jac=True, options={'disp': False, 'gtol': gtol}, args = ([M1_Y, M2_Y, M3_Y, list2, list3, 0, 0], full_psf, triplets, L, K, list2_all, list3_all))

# def optimize_2d_psf_expansion(initial_guesses, M1_Y, M2_Y, M3_Y, list2, list3, L, K, coeffs0=None, gtol=1e-10):
#     # Optimization using Fourier-Bessel expansion of the psf
#     if coeffs0 is None:
#         _, locations = psf_funcs.psf_initial_guess(L, initial_guesses[0])
#         num_of_coeffs = L
#         coeffs0 = psf_funcs.coeffs_initial_guess(L, locations, num_of_coeffs)
#     initial_guesses = np.concatenate((initial_guesses, coeffs0.flatten()))
#     list2_all = calclist2_all(L)
#     list3_all = calclist3_all(L)
#     # list3_all_reversed = [t[::-1] for t in list3_all]
#     return minimize(fun=c_g_funcs.cost_grad_fun_2d_psf_expansion, x0=initial_guesses, method='BFGS', jac=True, options={'disp': False, 'gtol': gtol}, args = ([M1_Y, M2_Y, M3_Y, list2, list3, 0, 0], L, K, list2_all, list3_all))

# def optimize_2d_known_psf_Adam(initial_guesses, M1_Y, M2_Y, M3_Y, list2, list3, full_psf, triplets, L, K, gtol=1e-10):
#     # Optimization assuming known psf
#     optizer = tf.compat.v1.train.AdamOptimizer()
#     initial_guesses = tf.Variable(initial_value=tf.convert_to_tensor(initial_guesses))
#     list2_all = calclist2_all(L)
#     list3_all = calclist3_all(L)
#     # list3_all_reversed = [t[::-1] for t in list3_all]
    
#     return optizer.minimize(tf.convert_to_tensor(c_g_funcs.cost_fun_2d_known_psf(initial_guesses, [M1_Y, M2_Y, M3_Y, list2, list3, 0, 0], full_psf, triplets, L, K, list2_all, list3_all)), var_list=initial_guesses)#([M1_Y, M2_Y, M3_Y, list2, list3, 0, 0], full_psf, triplets, L, K, list2_all, list3_all))
