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

from fb_funcs import expand_fb, rot_img, min_err_rots, min_err_coeffs, calc_jac, rot_img_freq, calcT, rot_img_freqT
from generate_clean_micrograph_2d import generate_clean_micrograph_2d_one_neighbor_rots, generate_clean_micrograph_2d_rots
from funcs_calc_moments import M2_2d, M3_2d
from funcs_calc_moments_rot import calck1k2k3, calcmap3, calck1k2k3_binned, calck1, calcS3_x, calcS3_x_neigh, calcS3_full_shift, calcS3_grad_full_shift, calcS3_x_grad, calcS3_x_grad_binned, calcS3_x_neigh_grad, calcS3_x_neigh_grad_binned, calcS2_grad_full_shift, calcS2_x_grad, calcS2_x_neigh_grad, calcN_mat
from psf_functions_2d import full_psf_2d, coeffs_initial_guess, calc_psf_from_coeffs
from tsf_functions_2d import full_tsf_2d
from costgrad import check_moments
import optimization_funcs_rot
import funcs_calc_moments_rot
from calc_estimation_error import calc_estimation_error
from generate_signal import generate_signal
from makeExtraMat import makeExtraMat
from maketsfMat import maketsfMat

from c_g_funcs_rot import calc_acs_grads_rot_parallel, calc_acs_grads_rot

import phantominator

plt.close("all")

if __name__ == '__main__':
    # script_dir = os.path.dirname(__file__)
    # rel_path = "images/molecule17.png"
    # file_path = os.path.join(script_dir, rel_path)
    # X = plt.imread("images/molecule17.png")
    # X = phantominator.ct_shepp_logan(17)
    X = np.random.rand(5, 5)
    # X = np.load('X_data.npy')
    L = np.shape(X)[0]
    X = X / np.linalg.norm(X)
    W = L # L for arbitrary spacing distribution, 2*L-1 for well-separated
    
    # X = generate_signal(2, L)
    
    # F = scipy.linalg.dft(L)
    # X = np.random.multivariate_normal(mean=np.zeros((L,)), cov=np.real(np.conjugate(np.transpose(F))*np.diag(np.array([5, 4, 3, 2, 1]))*F), size=(L, L))[:,:,0]
    # X = X - np.mean(X)
    # X = L*L * X / np.linalg.norm(X)
    # X = generate_signal(2, L)
    N = 1000
    ne = 50
    B, z, roots, kvals, nu = expand_fb(X, ne)
    T = calcT(nu, kvals)
    BT = B @ T.H
    c = np.real(T @ z)
    z = T.H@c
    Xrec = np.reshape(np.real(B @ z), np.shape(X))
    XrecT = np.reshape(np.real(BT @ c), np.shape(X))
    theta = np.random.uniform(0, 2*np.pi)
    # Xrot = rot_img(theta, z, kvals, B)
    # XrotT = rot_img(theta, c, kvals, BT)
    Bk = np.zeros((2*L-1, 2*L-1, nu), dtype=np.complex_)
    for i in range(nu):
        Bk[ :, :, i] = np.fft.fft2(np.pad(np.reshape(B[ :, i], (L, L)), L//2))
    
    Xrot_freq = rot_img_freq(theta, z, kvals, Bk, L)
    Xrot_freqT = rot_img_freqT(theta, c, kvals, Bk, L, T)
    y_clean, s, locs = generate_clean_micrograph_2d_rots(c, kvals, Bk, W, L, N, 0.10*(N/L)**2, T)
    # y_clean_triplets, s_triplets, locs_triplets = generate_clean_micrograph_2d_rots(c, kvals, Bk, W, L, N, 0.10*(N/L)**2, T)
    
    gamma = s[0]*(L/N)**2
    # gamma_tripltes = s_triplets[0]*(L/N)**2
    sigma2 = 0.0
    y = y_clean + np.random.default_rng().normal(loc=0, scale=np.sqrt(sigma2), size=np.shape(y_clean))
    # fig, axs = plt.subplots(2,2)
    # axs[0,0].imshow(X, cmap='gray')
    # axs[0,0].title.set_text('Original image')
    # axs[0,1].imshow(Xrec, cmap='gray')
    # axs[0,1].title.set_text('Reconstructed image')
    # axs[1,0].imshow(Xrot, cmap='gray')
    # axs[1,0].title.set_text('Rotated image')
    # axs[1,1].imshow(y_N, cmap='gray')
    # axs[1,1].title.set_text('A micrograph')
    # fig.suptitle(f'{ne} coefficients')
    
    kmax = np.max(kvals)
    
    # w_k_phi = Bk*np.exp(1j*kvals*theta)
    # Fk = w_k_phi@z
    # F = np.real(np.fft.ifft2(Fk))
    
    # s1, s2, s3 = check_moments(B, z, 6*kmax, 2*L-1, nu, kvals)
    # s2 = gamma*s2/(L**2)
    # s3 = gamma*s3/(L**2)
    
    # Xk = np.fft.fftshift(np.fft.fft2(Xrec))
    # Xcheck = np.real(np.fft.ifft2(np.fft.ifftshift(Xk)))
    # Xkz = np.fft.fftshift(np.reshape(Bk @ z, (2*L-1, 2*L-1)))
    # Xcheckz = np.real(np.fft.ifft2(np.fft.ifftshift(Xkz)))
    
    # fig, axs = plt.subplots(1,2)
    # axs[0].imshow(np.abs(Xk), cmap='gray')
    # axs[1].imshow(np.abs(Xkz), cmap='gray')
    
    yy = np.zeros((N, N, 1))
    yy[ :, :, 0] = y
    M1_y = np.mean(y)
    
    M2_y = np.zeros((L, L))
    for i1 in range(L):
        for j1 in range(L):
            M2_y[i1, j1] = M2_2d(yy, (i1, j1))
    
    # M2_y = np.zeros((2*L-1, 2*L-1))
    # for i1 in range(-(L-1), L):
    #     for j1 in range(-(L-1), L):
    #         M2_y[i1%(2*L-1), j1%(2*L-1)] = M2_2d(yy, (j1, i1))
            
    # A2_k = np.fft.fftn(M2_y)
    
    M3_y = np.zeros((L, L, L, L))
    for i1 in range(L):
        print(i1)
        for j1 in range(L):
            for i2 in range(L):
                for j2 in range(L):
                    M3_y[i1, j1, i2, j2] = M3_2d(yy, (i1, j1), (i2, j2))

    # M3_y = np.zeros((2*L-1, 2*L-1, 2*L-1, 2*L-1))
    # for i1 in range(-(L-1), L):
    #     print(i1)
    #     for j1 in range(-(L-1), L):
    #         for i2 in range(-(L-1), L):
    #             for j2 in range(-(L-1), L):
    #                 M3_y[i1%(2*L-1), j1%(2*L-1), i2%(2*L-1), j2%(2*L-1)] = M3_2d(yy, (j1, i1), (j2, i2))
    # A3_k = np.fft.fftn(M3_y)
    

    # S1_x = np.sum(np.fft.ifftn(Bk @ z), axis=(0, 1))/(L**2)
    # gS1_x = np.sum(np.fft.ifftn(Bk, axes=(0,1)), axis=(0, 1))/(L**2)                  
    # S3_k, gS3_k = funcs_calc_moments_rot.calcS3_x_grad_k(L, 6*np.max(kvals), Bk, z, kvals, calck1k2k3(L))
    # S3_x, gS3_x = funcs_calc_moments_rot.calcS3_x_grad(L, 6*np.max(kvals), Bk, z, kvals, calck1k2k3(L))
    # # diffz = 0.1*np.random.rand(len(z))
    # # S3_x1, gS3_x1 = funcs_calc_moments_rot.calcS3_x_grad(L, 6*np.max(kvals), Bk, z+diffz, kvals, calck1k2k3(L))
    # # diffS3 = S3_x1 - S3_x
    # S2_k, gS2_k = funcs_calc_moments_rot.calcS2_x_grad_k(L, 4*np.max(kvals), Bk, z, kvals, calck1(L))
    # S2_x, gS2_x = funcs_calc_moments_rot.calcS2_x_grad(L, 4*np.max(kvals), Bk, z, kvals, calck1(L))
    
    # objective_value = (1/2)*(M1_y - gamma*np.real(S1_x))**2 + (1/(2*L**2))*np.sum((M2_y - gamma*np.real(S2_x[ :L, :L]))**2) + (1/(2*L**4))*np.sum((M3_y - gamma*np.real(S3_x[ :L, :L, :L, :L]))**2)
    # import c_g_funcs_rot
    # # S2_x, gS2_x, S2_x_neigh, gS2_x_neigh, S3_x, gS3_x, S3_x_neigh, gS3_x_neigh = c_g_funcs_rot.calc_acs_grads_rot_parallel(Bk, z, kvals, L)                    
    
    # k1k2k3_map = calck1k2k3(L)
    # Nmax = 6*kmax
    # R = 1000
    # k1k2k3_map_binned, k1k2k3_map_equi_idxs = calck1k2k3_binned(L, R)
    # a = time.time()
    # S3_x, gS3_x = calcS3_x_grad(L, Nmax, Bk, z, kvals, k1k2k3_map)
    # print(f'Time passed, not binned :{time.time() - a}')
    # s3_x = S3_x.flatten()
    
    
    # a = time.time()
    # S3_x_binned, gS3_x_binned = calcS3_x_grad_binned(L, Nmax, Bk, z, kvals, k1k2k3_map_binned, k1k2k3_map_equi_idxs)
    # print(f'Time passed, binned :{time.time() - a}')
    # s3_x_binned = S3_x_binned.flatten()
    
    
    # z0 = z[kvals==0]
    # Nmax_neigh = 4*kmax
    # a = time.time()
    # S3_x_neigh, gS3_x_neigh = calcS3_x_neigh_grad(L, Nmax_neigh, Bk, z, kvals, k1k2k3_map)
    # print(f'Time passed, not binned :{time.time() - a}')
    
    # a = time.time()
    # S3_x_neigh_binned, gS3_x_neigh_binned = calcS3_x_neigh_grad_binned(L, Nmax, Bk, z, kvals, k1k2k3_map_binned, k1k2k3_map_equi_idxs)
    # print(f'Time passed, binned :{time.time() - a}')
    
    
    # k1_map = calck1(L)
    # Nmax = 4*kmax
    # S2_x, gS2_x = calcS2_x_grad(L, Nmax, Bk, z, kvals, k1_map)
    
    # S2_x_neigh, gS2_x_neigh = calcS2_x_neigh_grad(L, Bk, z, kvals, k1_map)
    
    

    psf = full_psf_2d(locs, L)
    tsf = full_tsf_2d(locs, L)

    # N_mat = calcN_mat(L)
    
    # ExtraMat = makeExtraMat(L, psf)
    # # ExtraMat = scipy.sparse.csr_matrix(ExtraMat)
    
    # M2_y_check = np.zeros((L, L), dtype=np.complex_)
    # for i1 in range(L):
    #     print(i1)
    #     for j1 in range(L):
    #         for i2 in range(L):
    #             for j2 in range(L):
    #                 M2_y_check[i1, j1], _ = calcS2_grad_full_shift((j1, i1), S2_x, gS2_x, S2_x_neigh, gS2_x_neigh, L, psf) 
                    
    # M2_y_check = gamma*np.real(M2_y_check)
    # ratio2 = (M2_y/M2_y_check).flatten()
    
    # M3_y_check = np.zeros((L, L, L, L), dtype=np.complex_)
    # gM3_y_check = np.zeros((L, L, L, L, len(z)), dtype=np.complex_)
    # for i1 in range(L):
    #     print(i1)
    #     for j1 in range(L):
    #         for i2 in range(L):
    #             for j2 in range(L):
    #                 M3_y_check[i1, j1, i2, j2], gM3_y_check[i1, j1, i2, j2, :] = calcS3_grad_full_shift((j1, i1), (j2, i2), S3_x, gS3_x, S3_x_neigh, gS3_x_neigh, L, psf) 
                    
    # M3_y_check = gamma*M3_y_check
    # ratio3 = (M3_y/M3_y_check).flatten()
    # print('success')
    # ExtraMat = makeExtraMat(L, psf)
    # print('Done')
    
    gamma_initial = 0.090
    # y_init, s_init, locs_init = generate_clean_micrograph_2d_one_neighbor_rots(c, kvals, Bk, W, L, N, gamma_initial*(N/L)**2, T)
    # psf_init = full_psf_2d(locs_init, L)
    # ExtraMat2, ExtraMat3 = makeExtraMat(L, psf)
    # tsfMat = maketsfMat(L, tsf)
    # print('Done')
    # psf_coeffs_initial = coeffs_initial_guess(L, locs_init, 20)
    # F = scipy.linalg.dft(L)
    # X_initial = np.random.multivariate_normal(mean=np.zeros((L,)), cov=np.real(np.conjugate(np.transpose(F))*np.diag(np.array([5, 4, 3, 2, 1]))*F), size=(L, L))[:,:,0]
    # X_initial = X_initial - np.mean(X_initial)
    # X_initial = generate_signal(2, L)
    X_initial = np.random.rand(L, L) # X + np.random.normal(loc=0, scale=0.05, size=np.shape(X))
    X_initial = X_initial / np.linalg.norm(X_initial)
    
    # X_initial = L*L * X_initial / np.linalg.norm(X_initial)
    _, z_initial, _, _, _ = expand_fb(X_initial, ne)
    c_initial = np.real(T @ z_initial)
    # start = time.time()
    # X_est_binned = optimize_2d_known_psf_mat_rot_binned(np.concatenate((np.reshape(gamma_initial, (1,)), np.real(z_initial), np.imag(z_initial))), Bk, kvals, M1_y, M2_y, M3_y, sigma2, psf_init, L, 1, R, gtol=1e-6)
    # time_passed = time.time() - start
    # print(f'Time passed: {time_passed} secs')
    # gamma_est_binned = X_est_binned.x[0]
    # z_est_binned = X_est_binned.x[1:nu+1] + 1j*X_est_binned.x[nu+1:]
    # X_estimated_binned = np.reshape(np.real(B @ z_est_binned), np.shape(X))
    # est_err_binned = calc_estimation_error(Xrec, X_estimated_binned)
    # z_initial = z + np.random.rand(len(z))/10 + 1j*np.random.rand(len(z))/10    
    
    
    # # %% Initial psf for all run
    # start = time.time()
    # # X_est = optimization_funcs_rot.optimize_2d_known_psf_rot_parallel(np.concatenate((np.reshape(gamma_initial, (1,)), np.real(z_initial), np.imag(z_initial))), Bk, kvals, M1_y, M2_y, M3_y, sigma2, psf, L, 1, gtol=1e-13) # np.reshape(gamma_initial, (1,)),
    
    
    # # X_est_init_psf = optimization_funcs_rot.optimize_2d_known_psf_mat_rot_gamma(np.concatenate((np.reshape(gamma_initial, (1,)), c_initial)), Bk, T, kvals, M1_y, M2_y, M3_y, sigma2, psf_init, L, 1, ExtraMat2, ExtraMat3, numiters=3000, gtol=1e-15) # np.reshape(gamma_initial, (1,)),
    # X_est_init_psf = optimization_funcs_rot.optimize_2d_known_psf_triplets(np.concatenate((np.reshape(gamma_initial, (1,)), c_initial)), Bk, T, kvals, M1_y, M2_y, M3_y, sigma2, psf, L, 1, tsfMat, ExtraMat2, ExtraMat3, numiters=3000, gtol=1e-15) # np.reshape(gamma_initial, (1,)),
    
    # time_passed = time.time() - start
    # print(f'Time passed: {time_passed} secs')    
    # gamma_est_init_psf = X_est_init_psf.x[0] 
    # c_est_init_psf = X_est_init_psf.x[1:]
    # z_est_init_psf = T.H@c_est_init_psf
    # X_estimated_init_psf = np.reshape(np.real(B @ z_est_init_psf), np.shape(X))
    # est_err_init_psf = min_err_rots(z, z_est_init_psf, kvals, B, L)
    # est_err_coeffs_init_psf = min_err_coeffs(z, z_est_init_psf, kvals)
    
    # print(f'Final objective function value: {X_est_init_psf.fun}. Estimation error: {est_err_coeffs_init_psf[0] * 100}% for rotation of {est_err_coeffs_init_psf[1]} radians.')
    
    # X_est_rotated_init_psf = rot_img(est_err_coeffs_init_psf[1], z_est_init_psf, kvals, B)
    
    # plt.figure()
    # plt.subplot(1,2,1)
    # plt.imshow(Xrec)
    # plt.title('Original')
    # plt.xticks([])
    # plt.yticks([])
    # plt.subplot(1,2,2)
    # plt.imshow(X_est_rotated_init_psf)
    # plt.title('Estimated')
    # plt.xticks([])
    # plt.yticks([])
    
    # # %% Iterate over psf
    # start = time.time()
    
    # X_est, psf_est = optimization_funcs_rot.optimize_2d_x_psf(np.concatenate((np.reshape(gamma_initial, (1,)), c_initial)), Bk, T, kvals, M1_y, M2_y, M3_y, sigma2, psf_init, L, 1) # np.reshape(gamma_initial, (1,)),
    # psf_estimated = np.reshape(psf_est.x, (4*L-3, 4*L-3))
    
    # # X_est = optimization_funcs_rot.optimize_2d_known_psf_rot_k(np.concatenate((np.reshape(gamma_initial, (1,)), np.real(z_initial), np.imag(z_initial))), Bk, kvals, M1_y, A2_k, A3_k, sigma2, psf, L, 1, gtol=1e-13)#, ExtraMat, gtol=1e-13) # np.reshape(gamma_initial, (1,)),

    # time_passed = time.time() - start
    # print(f'Time passed: {time_passed} secs')
    # gamma_est = X_est.x[0] 
    # c_est = X_est.x[1:]
    # z_est = T.H@c_est
    # X_estimated = np.reshape(np.real(B @ z_est), np.shape(X))
    # est_err = min_err_rots(z, z_est, kvals, B, L)
    # est_err_coeffs = min_err_coeffs(z, z_est, kvals)
    
    # print(f'Final objective function value: {X_est.fun}. Estimation error: {est_err_coeffs[0] * 100}% for rotation of {est_err_coeffs[1]} radians.')
    
    # X_est_rotated = rot_img(est_err_coeffs[1], z_est, kvals, B)
    
    # plt.figure()
    # plt.subplot(1,2,1)
    # plt.imshow(Xrec)
    # plt.title('Original')
    # plt.xticks([])
    # plt.yticks([])
    # plt.subplot(1,2,2)
    # plt.imshow(X_est_rotated)
    # plt.title('Estimated')
    # plt.xticks([])
    # plt.yticks([])
    
    
    
    # # %% Change psf
    # start = time.time()
    
    # X_est_change, psf_estimated_change = optimization_funcs_rot.optimize_2d_x_update_psf(np.concatenate((np.reshape(gamma_initial, (1,)), c_initial)), Bk, T, kvals, M1_y, M2_y, M3_y, sigma2, L, 1, W, N, iters_till_change=300, gtol=1e-15)

    # time_passed = time.time() - start
    # print(f'Time passed: {time_passed} secs')
    # gamma_est_change = X_est_change.x[0] 
    # c_est_change = X_est_change.x[1:]
    # z_est_change = T.H@c_est_change

    # X_estimated_change = np.reshape(np.real(B @ z_est_change), np.shape(X))
    # est_err_change = min_err_rots(z, z_est_change, kvals, B, L)
    # est_err_coeffs_change = min_err_coeffs(z, z_est_change, kvals)
    # z_est_change_best = z_est_change * np.exp(1j*kvals*est_err_coeffs_change[1])
    
    # print(f'Final objective function value: {X_est_change.fun}. Estimation error: {est_err_coeffs_change[0] * 100}% for rotation of {est_err_coeffs_change[1]} radians.')
    
    # X_est_rotated_change = rot_img(est_err_coeffs_change[1], z_est_change, kvals, B)
    
    # plt.figure()
    # plt.subplot(1,2,1)
    # plt.imshow(Xrec, cmap='gray')
    # plt.title('Original')
    # plt.xticks([])
    # plt.yticks([])
    # plt.subplot(1,2,2)
    # plt.imshow(X_est_rotated_change, cmap='gray')
    # plt.title('Estimated')
    # plt.xticks([])
    # plt.yticks([])
    
    # %% Change psf and tsf
    start = time.time()
    
    X_est_change, psf_estimated_change, tsf_estimated_change = optimization_funcs_rot.optimize_rot_Algorithm1_notparallel(np.concatenate((np.reshape(gamma_initial, (1,)), c_initial)), Bk, T, kvals, M1_y, M2_y, M3_y, sigma2, L, 1, W, N, iters_till_change=150, gtol=1e-15)

    time_passed = time.time() - start
    print(f'Time passed: {time_passed} secs')
    gamma_est_change = X_est_change.x[0] 
    c_est_change = X_est_change.x[1:]
    z_est_change = T.H@c_est_change

    X_estimated_change = np.reshape(np.real(B @ z_est_change), np.shape(X))
    est_err_change = min_err_rots(z, z_est_change, kvals, B, L)
    est_err_coeffs_change = min_err_coeffs(z, z_est_change, kvals)
    z_est_change_best = z_est_change * np.exp(1j*kvals*est_err_coeffs_change[1])
    
    print(f'Final objective function value: {X_est_change.fun}. Estimation error: {est_err_coeffs_change[0] * 100}% for rotation of {est_err_coeffs_change[1]} radians.')
    
    X_est_rotated_change = rot_img(est_err_coeffs_change[1], z_est_change, kvals, B)
    
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
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    # plt.figure()
    # plt.plot(ratio)
    
#GOOD
    # S1 = np.real(np.sum(np.fft.ifftn(Bk @ z_est), axis=(0, 1))/(L**2))    
    # S2_x, _, S2_x_neigh, _, S3_x, _, S3_x_neigh, _ = calc_acs_grads_rot(Bk, z_est, kvals, L) ### GIVE THE FUNCTION THE MAPS
    # S3_x = np.real(S3_x)
    # S3_x_neigh = np.real(S3_x_neigh)
    
    # psf_est1 = optimization_funcs_rot.optimize_2d_psf(psf_init.flatten(), gamma_est, S1, S2_x, S2_x_neigh, S3_x, S3_x_neigh, Bk, kvals, M1_y, M2_y, M3_y, sigma2, L, 1, gtol=1e-15)
    # psf_new1 = np.reshape(psf_est1.x, (4*L-3, 4*L-3))
    # ExtraMat2_new1, ExtraMat3_new1 = makeExtraMat(L, psf_new1)

    # X_est_new1 = optimization_funcs_rot.optimize_2d_known_psf_mat_rot_gamma(np.concatenate((np.reshape(gamma_est, (1,)), c_est)), Bk, T, kvals, M1_y, M2_y, M3_y, sigma2, psf_new1, L, 1, ExtraMat2_new1, ExtraMat3_new1, numiters=150, gtol=1e-15) # np.reshape(gamma_initial, (1,)),

    # gamma_est_new1 = X_est_new1.x[0] 
    # c_est_new1 = X_est_new1.x[1:]
    # z_est_new1 = T.H@c_est_new1
    # S1 = np.real(np.sum(np.fft.ifftn(Bk @ z_est_new1), axis=(0, 1))/(L**2))    
    # S2_x, _, S2_x_neigh, _, S3_x, _, S3_x_neigh, _ = calc_acs_grads_rot(Bk, z_est_new1, kvals, L) ### GIVE THE FUNCTION THE MAPS
    # S3_x = np.real(S3_x)
    # S3_x_neigh = np.real(S3_x_neigh)
    
    # psf_est2 = optimization_funcs_rot.optimize_2d_psf(psf_new1.flatten(), gamma_est_new1, S1, S2_x, S2_x_neigh, S3_x, S3_x_neigh, Bk, kvals, M1_y, M2_y, M3_y, sigma2, L, 1, gtol=1e-15)
    # psf_new2 = np.reshape(psf_est2.x, (4*L-3, 4*L-3))
    # ExtraMat2_new2, ExtraMat3_new2 = makeExtraMat(L, psf_new2)
    
    # X_est_new2 = optimization_funcs_rot.optimize_2d_known_psf_mat_rot_gamma(np.concatenate((np.reshape(gamma_est_new1, (1,)), c_est_new1)), Bk, T, kvals, M1_y, M2_y, M3_y, sigma2, psf_new2, L, 1, ExtraMat2_new2, ExtraMat3_new2, numiters=2000, gtol=1e-15) # np.reshape(gamma_initial, (1,)),

    # gamma_est_new2 = X_est_new2.x[0] 
    # c_est_new2 = X_est_new2.x[1:]
    # z_est_new2 = T.H@c_est_new2
    # GOOD
    
    # S1 = np.real(np.sum(np.fft.ifftn(Bk @ z_est_new2), axis=(0, 1))/(L**2))    
    # S2_x, _, S2_x_neigh, _, S3_x, _, S3_x_neigh, _ = calc_acs_grads_rot(Bk, z_est_new2, kvals, L) ### GIVE THE FUNCTION THE MAPS
    # S3_x = np.real(S3_x)
    # S3_x_neigh = np.real(S3_x_neigh)
    
    # psf_est3 = optimization_funcs_rot.optimize_2d_psf(psf_new2.flatten(), gamma_est_new2, S1, S2_x, S2_x_neigh, S3_x, S3_x_neigh, Bk, kvals, M1_y, M2_y, M3_y, sigma2, L, 1, numiters=500, gtol=1e-15)
    # psf_new3 = np.reshape(psf_est3.x, (4*L-3, 4*L-3))
    # ExtraMat2_new3, ExtraMat3_new3 = makeExtraMat(L, psf_new3)
    
    # X_est_new3 = optimization_funcs_rot.optimize_2d_known_psf_mat_rot_gamma(np.concatenate((np.reshape(gamma_est_new2, (1,)), c_est_new2)), Bk, T, kvals, M1_y, M2_y, M3_y, sigma2, psf_new3, L, 1, ExtraMat2_new3, ExtraMat3_new3, numiters=1000, gtol=1e-15) # np.reshape(gamma_initial, (1,)),

    # gamma_est_new3 = X_est_new3.x[0] 
    # c_est_new3 = X_est_new3.x[1:]
    # z_est_new3 = T.H@c_est_new3
    
    
    

# GOOD
    # X_estimated_new2 = np.reshape(np.real(B @ z_est_new2), np.shape(X))
    # est_err = min_err_rots(z, z_est_new2, kvals, B, L)
    # est_err_coeffs = min_err_coeffs(z, z_est_new2, kvals)
    
    # print(f'Final objective function value: {X_est_new2.fun}. Estimation error: {est_err_coeffs[0] * 100}% for rotation of {est_err_coeffs[1]} radians.')
    
    # X_est_rotated = rot_img(est_err_coeffs[1], z_est_new2, kvals, B)
    
    # plt.figure()
    # plt.subplot(1,2,1)
    # plt.imshow(Xrec)
    # plt.title('Original')
    # plt.xticks([])
    # plt.yticks([])
    # plt.subplot(1,2,2)
    # plt.imshow(X_est_rotated)
    # plt.title('Estimated')
    # plt.xticks([])
    # plt.yticks([])
    # GOOD
    # ## Now do the psf
    # _, _, _, _, S3_x, _, S3_x_neigh, _ = calc_acs_grads_rot_parallel(Bk, z_est, kvals, L, calck1(L), calck1k2k3(L))

    # start = time.time()
    # coeffs_estimation = optimization_funcs_rot.optimize_2d_rot_psf_coeffs(psf_coeffs_initial, gamma_est, np.mean(X_estimated), S3_x, S3_x_neigh, Bk, kvals, M1_y, M2_y, M3_y, sigma2, L, 1)
    # time_passed_coeffs = time.time() - start
    # coeffs_estimated = coeffs_estimation.x
    # psf_new = calc_psf_from_coeffs(coeffs_estimated, L)
    
    # ## And again X and gamma
    # y_new, _, locs_new = generate_clean_micrograph_2d_one_neighbor_rots(z, kvals, Bk, L, L, N, (gamma_est*(N/L)**2).astype(int))
    # psf_new = full_psf_2d(locs_new, L)
    # start = time.time()
    # X_est_new = optimization_funcs_rot.optimize_2d_known_psf_mat_rot_gamma_parallel(np.concatenate((np.reshape(gamma_est, (1,)), np.real(z_est), np.imag(z_est))), Bk, kvals, M1_y, M2_y, M3_y, sigma2, psf_new, L, 1, ExtraMat, gtol=1e-13) # np.reshape(gamma_initial, (1,)),
    # time_passed_new = time.time() - start
    
    # gamma_est_new = X_est_new.x[0] 
    # z_est_new = X_est_new.x[1:nu+1] + 1j*X_est_new.x[nu+1:]
    # X_estimated_new = np.reshape(np.real(B @ z_est_new), np.shape(X))
    # est_err_new = min_err_coeffs(z, z_est_new, kvals)
    
    # print(f'Final objective function value: {X_est.fun}. Estimation error: {est_err[0]} for rotation of {est_err[1]} radians.')
    
    # X_est_rotated = rot_img(est_err[1], z_est, kvals, B)
    
    # plt.figure()
    # plt.subplot(1,2,1)
    # plt.imshow(Xrec)
    # plt.title('Original')
    # plt.xticks([])
    # plt.yticks([])
    # plt.subplot(1,2,2)
    # plt.imshow(X_est_rotated)
    # plt.title('Estimated')
    # plt.xticks([])
    # plt.yticks([]) 
    
    
    
    # X_check = np.real(np.reshape(B@z_est, (L, L)))

    # S3_x_check, gS3_x_check = funcs_calc_moments_rot.calcS3_x_grad(L, 6*np.max(kvals), Bk, z_est, kvals, calck1k2k3(L))
    # ratio3 = M3_y[3,1, :, :] / (gamma_est*S3_x_check[3, 1, :L, :L])
    # S2_x_check, gS2_x_check = funcs_calc_moments_rot.calcS2_x_grad(L, 4*np.max(kvals), Bk, z_est, kvals, calck1(L))
    # ratio2_est = M2_y / (gamma_est*S2_x_check[ :L, :L])
    # ratio2 = M2_y / (gamma*S2_x[ :L, :L])
    
    # S3_x_initial, gS3_x_initial = funcs_calc_moments_rot.calcS3_x_grad(L, 6*np.max(kvals), Bk, z_initial, kvals, calck1k2k3(L))

    # S2_x_initial, gS2_x_initial = funcs_calc_moments_rot.calcS2_x_grad(L, 4*np.max(kvals), Bk, z_initial, kvals, calck1(L))
    
    