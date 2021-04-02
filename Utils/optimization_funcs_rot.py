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
from funcs_calc_moments_rot import calcmap3, calck1, calcN_mat
from makeExtraMat import makeExtraMat
from maketsfMat import maketsfMat
from maketsfMat_parallel import maketsfMat_parallel
import scipy.special as special
from generate_clean_micrograph_2d import generate_clean_micrograph_2d_one_neighbor_rots, generate_clean_micrograph_2d_rots

import psf_functions_2d
import tsf_functions_2d

def optimize_2d_known_psf_triplets(initial_guesses, Bk, T, kvals, M1_y, M2_y, M3_y, sigma2, L, K, tsfMat, ExtraMat2, ExtraMat3, numiters=3000, gtol=1e-15):
    # Optimization assuming known psf and tsf, and all matrices already computed
    k1_map = calck1(L)
    map3 = calcmap3(L)
    N_mat = calcN_mat(L)
    
    return minimize(fun=c_g_funcs_rot.cost_grad_fun_rot_notparallel, x0=initial_guesses, method='BFGS', jac=True, options={'disp': True, 'maxiter':numiters, 'gtol': gtol}, args = (Bk, T, kvals, M1_y, M2_y, M3_y, sigma2, ExtraMat2, ExtraMat3, tsfMat, L, K, N_mat, k1_map, map3))

def optimize_rot_Algorithm1_parallel(initial_guesses, Bk, T, kvals, M1_y, M2_y, M3_y, sigma2, L, K, W, N, iters_till_change=150, gtol=1e-15):
    k1_map = calck1(L)
    map3 = calcmap3(L)
    N_mat = calcN_mat(L)
    
    y_init, _, locs_init = generate_clean_micrograph_2d_rots(initial_guesses[1: ], kvals, Bk, W, L, 10000, (initial_guesses[0]*(10000/L)**2).astype(int), T)
    psf_init = psf_functions_2d.full_psf_2d(locs_init, L)
    ExtraMat2, ExtraMat3 = makeExtraMat(L, psf_init)
    tsf_init = tsf_functions_2d.full_tsf_2d(locs_init, L)
    tsfMat = maketsfMat_parallel(L, tsf_init)
        
    first_estimates = minimize(fun=c_g_funcs_rot.cost_grad_fun_rot_parallel, x0=initial_guesses, method='BFGS', jac=True, options={'disp': True, 'maxiter':iters_till_change, 'gtol': gtol}, args = (Bk, T, kvals, M1_y, M2_y, M3_y, sigma2, ExtraMat2, ExtraMat3, tsfMat, L, K, N_mat, k1_map, map3))
    first_gamma = first_estimates.x[0]
    print(f'We got to gamma of {first_gamma}')
    first_c = first_estimates.x[1: ]
    
    y2, _, locs2 = generate_clean_micrograph_2d_rots(first_c, kvals, Bk, W, L, N, (first_gamma*(N/L)**2).astype(int), T)
    psf2 = psf_functions_2d.full_psf_2d(locs2, L)
    ExtraMat2, ExtraMat3 = makeExtraMat(L, psf2)
    tsf2 = tsf_functions_2d.full_tsf_2d(locs2, L)
    tsfMat = maketsfMat_parallel(L, tsf2)
    
    new_guesses = np.concatenate((np.reshape(first_gamma, (1,)), first_c))
    
    return minimize(fun=c_g_funcs_rot.cost_grad_fun_rot_parallel, x0=new_guesses, method='BFGS', jac=True, options={'disp': True, 'maxiter':2000, 'gtol': gtol}, args = (Bk, T, kvals, M1_y, M2_y, M3_y, sigma2, ExtraMat2, ExtraMat3, tsfMat, L, K, N_mat, k1_map, map3)), psf2, tsf2
   
def optimize_rot_Algorithm1_notparallel(initial_guesses, Bk, T, kvals, M1_y, M2_y, M3_y, sigma2, L, K, W, N, iters_till_change=150, gtol=1e-15):
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
        
    first_estimates = minimize(fun=c_g_funcs_rot.cost_grad_fun_rot_notparallel, x0=initial_guesses, method='BFGS', jac=True, options={'disp': True, 'maxiter':iters_till_change, 'gtol': gtol}, args = (Bk, T, kvals, M1_y, M2_y, M3_y, sigma2, ExtraMat2, ExtraMat3, tsfMat, L, K, N_mat, k1_map, map3))
    first_gamma = first_estimates.x[0]
    print(f'We got to gamma of {first_gamma}')
    first_c = first_estimates.x[1: ]
    
    y2, _, locs2 = generate_clean_micrograph_2d_rots(first_c, kvals, Bk, W, L, 10000, (first_gamma*(10000/L)**2).astype(int), T)
    psf2 = psf_functions_2d.full_psf_2d(locs2, L)
    ExtraMat2, ExtraMat3 = makeExtraMat(L, psf2)
    tsf2 = tsf_functions_2d.full_tsf_2d(locs2, L)
    tsfMat = maketsfMat(L, tsf2)
    
    new_guesses = np.concatenate((np.reshape(first_gamma, (1,)), first_c))
    
    return minimize(fun=c_g_funcs_rot.cost_grad_fun_rot_notparallel, x0=new_guesses, method='BFGS', jac=True, options={'disp': True, 'maxiter':2000, 'gtol': gtol}, args = (Bk, T, kvals, M1_y, M2_y, M3_y, sigma2, ExtraMat2, ExtraMat3, tsfMat, L, K, N_mat, k1_map, map3)), psf2, tsf2
