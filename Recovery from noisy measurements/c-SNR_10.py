# -*- coding: utf-8 -*-
"""
Created on Sat Apr  3 17:33:40 2021

@author: kreym
"""

import numpy as np
import matplotlib.pyplot as plt

import random

import time


from Utils.fb_funcs import expand_fb, rot_img, min_err_rots, min_err_coeffs, calcT
from Utils.generate_clean_micrograph_2d import generate_clean_micrograph_2d_one_neighbor_rots, generate_clean_micrograph_2d_rots
import Utils.optimization_funcs_rot

from Utils.calcM3_parallel import calcM3_parallel_micrographs, calcM3_parallel_shifts

plt.close("all")
np.random.seed(100)
if __name__ == '__main__':
    X = plt.imread("../images/molecule9.png")
    X = X * 10
    L = np.shape(X)[0]
    W = L # L for arbitrary spacing distribution, 2*L-1 for well-separated

    N = 50000
    # NumMicrographs = 1
    ne = 34
    B, z, roots, kvals, nu = expand_fb(X, ne)
    T = calcT(nu, kvals)
    BT = B @ T.H
    c = np.real(T @ z)
    z = T.H@c
    Xrec = np.reshape(np.real(B @ z), np.shape(X))

    Bk = np.zeros((2*L-1, 2*L-1, nu), dtype=np.complex_)
    for i in range(nu):
        Bk[ :, :, i] = np.fft.fft2(np.pad(np.reshape(B[ :, i], (L, L)), L//2))
    
    gamma = 0.1
    SNR = 10
    sigma2 = np.linalg.norm(Xrec)**2 / (SNR * np.pi * (L//2)**2)
    
    t_acs_start = time.time()
    M1_y, M2_y, M3_y, y = calcM3_parallel_shifts(L, sigma2, gamma, c, kvals, Bk, W, T, N)
    t_acs_finish = time.time() - t_acs_start
    
    np.save('../Results/Recovery/M1_y_9_SNR10.npy', M1_y)
    
    np.save('../Results/Recovery/M2_y_9_SNR10.npy', M2_y)
    
    np.save('../Results/Recovery/M3_y_9_SNR10.npy', M3_y)
    
    print("all saved")
    
    # M1_y = np.load('Results/Recovery/M1_ys_9_SNRinf.npy')
    
    # M2_y = np.load('Results/Recovery/M2_ys_9_SNRinf.npy')
    
    # M3_y = np.load('Results/Recovery/M3_ys_9_SNRinf.npy')
    
    # print("all loaded")
    
    gamma_initial = 0.09
    X_initial = np.random.rand(L, L)
    X_initial = np.linalg.norm(Xrec) * X_initial / np.linalg.norm(X_initial)
    
    _, z_initial, _, _, _ = expand_fb(X_initial, ne)
    c_initial = np.real(T @ z_initial)
    
    start = time.time()
    
    X_est, psf_estimated, tsf_estimated = Utils.optimization_funcs_rot.optimize_rot_Algorithm1_parallel(np.concatenate((np.reshape(gamma_initial, (1,)), c_initial)), Bk, T, kvals, M1_y, M2_y, M3_y, sigma2, L, 1, W, N, iters_till_change=100, gtol=1e-15, max_iter=2500)

    time_passed = time.time() - start
    print(f'Time passed: {time_passed} secs')
    
    gamma_est = X_est.x[0] 
    c_est = X_est.x[1:]
    z_est = T.H@c_est

    X_estimated_change = np.reshape(np.real(B @ z_est), np.shape(X))
    # est_err_change = min_err_rots(z, z_est, kvals, B, L)
    est_err_coeffs_change = min_err_coeffs(z, z_est, kvals)
    z_est_best = z_est * np.exp(1j*kvals*est_err_coeffs_change[1])
    
    print(f'Final objective function value: {X_est.fun}. Estimation error: {est_err_coeffs_change[0] * 100}% for rotation of {est_err_coeffs_change[1]} radians.')
    
    X_est_rotated_change = rot_img(est_err_coeffs_change[1], z_est, kvals, B)
    
    plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(Xrec, cmap='gray')
    plt.title('Original')
    plt.xticks([])
    plt.yticks([])
    plt.subplot(1,2,2)
    plt.imshow(X_est_rotated_change, cmap='gray')
    plt.title('Estimated')
    plt.xticks([])
    plt.yticks([])
    