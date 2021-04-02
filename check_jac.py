# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 20:26:09 2020

@author: kreym
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import scipy
import time

import multiprocessing as mp

from fb_funcs import expand_fb, rot_img, min_err_rots, min_err_coeffs, calc_jac
from generate_clean_micrograph_2d import generate_clean_micrograph_2d_one_neighbor_rots
from funcs_calc_moments import M2_2d, M3_2d
from funcs_calc_moments_rot import calck1k2k3, calck1k2k3_binned, calck1, calcS3_x, calcS3_x_neigh, calcS3_full_shift, calcS3_grad_full_shift, calcS3_x_grad, calcS3_x_grad_binned, calcS3_x_neigh_grad, calcS3_x_neigh_grad_binned, calcS2_grad_full_shift, calcS2_x_grad, calcS2_x_neigh_grad, calcN_mat
from psf_functions_2d import full_psf_2d
from costgrad import check_moments
import optimization_funcs_rot
from calc_estimation_error import calc_estimation_error
from generate_signal import generate_signal
from makeExtraMat import makeExtraMat
plt.close("all")

if __name__ == '__main__':
    # script_dir = os.path.dirname(__file__)
    # rel_path = "images/tiger17.png"
    # file_path = os.path.join(script_dir, rel_path)
    # X = plt.imread(file_path)99999999
    X = np.random.rand(9, 9)
    # X = np.load('X_data.npy')
    L = np.shape(X)[0]
    X = X / np.linalg.norm(X)
    
    # X = generate_signal(2, L)
    
    # F = scipy.linalg.dft(L)
    # X = np.random.multivariate_normal(mean=np.zeros((L,)), cov=np.real(np.conjugate(np.transpose(F))*np.diag(np.array([5, 4, 3, 2, 1]))*F), size=(L, L))[:,:,0]
    # X = X - np.mean(X)
    # X = L*L * X / np.linalg.norm(X)
    # X = generate_signal(2, L)
    N = 1000
    ne = 100
    B, z, roots, kvals, nu = expand_fb(X, ne)
    Xrec = np.reshape(np.real(B @ z), np.shape(X))
    theta = np.random.uniform(0, 2*np.pi)
    Xrot = rot_img(theta, z, kvals, B)
    y_clean, s, locs = generate_clean_micrograph_2d_one_neighbor_rots(z, kvals, B, L, N, 0.10*(N/L)**2)
    gamma = s[0]*(L/N)**2
    sigma2 = 0
    y = y_clean + np.random.default_rng().normal(loc=0, scale=np.sqrt(sigma2), size=np.shape(y_clean))
    
    Bk = np.zeros((2*L-1, 2*L-1, nu), dtype=np.complex_)
    for i in range(nu):
        Bk[ :, :, i] = np.fft.fft2(np.pad(np.reshape(B[ :, i], (L, L)), L//2))
        
    k1k2k3_map = calck1k2k3(L)

    
    k1_map = calck1(L)

    

    psf = full_psf_2d(locs, L)
    N_mat = calcN_mat(L)
    

    ExtraMat = makeExtraMat(L, psf)
    
    gamma_initial = 0.095
    y_init, s_init, locs_init = generate_clean_micrograph_2d_one_neighbor_rots(z, kvals, B, L, N, gamma_initial*(N/L)**2)
    psf_init = full_psf_2d(locs_init, L)
    
    X_initial = np.random.rand(L, L) # X + np.random.normal(loc=0, scale=0.05, size=np.shape(X))
    X_initial = X_initial / np.linalg.norm(X_initial)

    _, z_initial, _, _, _ = expand_fb(X_initial, ne)

    jac = calc_jac(np.concatenate((np.reshape(gamma, (1,)), np.real(z), np.imag(z))), Bk, kvals, sigma2, ExtraMat, psf, L, 1, N_mat, k1_map, k1k2k3_map)
    
    jac_cond = np.linalg.cond(jac)
    
    print(f'Condition number: {jac_cond}')