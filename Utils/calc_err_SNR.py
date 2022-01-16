# -*- coding: utf-8 -*-
"""
Created on Sun Jan 24 18:38:45 2021

@author: Shay Kreymer
"""

import numpy as np
from scipy import signal
import photutils.detection
import time
from Utils.fb_funcs import expand_fb, min_err_coeffs, calcT, symmetric_target
from Utils.generate_clean_micrograph_2d import generate_clean_micrograph_2d_rots
from Utils.funcs_calc_moments import M2_2d, M3_2d
from Utils.psf_tsf_funcs import full_psf_2d, full_tsf_2d, makeExtraMat, maketsfMat
from Utils.calcM3_parallel import calcM3_parallel_micrographs_given
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
        errs_known: an array containing the estimation errors for each size, known separation functions 
        costs_known: an array containing the objective function values for each size, known separation functions
        errs_Algorithm1: an array containing the estimation errors for each size, Algorithm 1
        costs_Algorithm1: an array containing the objective function values for each size, Algorithm 1
    """
    # %% preliminary definitions
    np.random.seed(sd)
    NumGuesses = 3
    errs_known = np.zeros((len(SNRs), NumGuesses))
    costs_known = np.zeros((len(SNRs), NumGuesses))
    errs_Algorithm1 = np.zeros((len(SNRs), NumGuesses))
    costs_Algorithm1 = np.zeros((len(SNRs), NumGuesses))
    X = np.random.rand(L, L)
    X = X / np.linalg.norm(X)

    W = L  # L for arbitrary spacing distribution, 2*L-1 for well-separated

    B, z, roots, kvals, nu = expand_fb(X, ne)
    T = calcT(nu, kvals)
    c = np.real(T @ z)
    z = T.H @ c
    Bk = np.zeros((2 * L - 1, 2 * L - 1, nu), dtype=np.complex_)
    for ii in range(nu):
        Bk[:, :, ii] = np.fft.fft2(np.pad(np.reshape(B[:, ii], (L, L)), L // 2))

    Xrec = np.reshape(np.real(B @ z), np.shape(X))
    Xrec = Xrec / np.linalg.norm(Xrec)
    B, z, roots, kvals, nu = expand_fb(Xrec, ne)
    T = calcT(nu, kvals)
    c = np.real(T @ z)
    z = T.H @ c
    Bk = np.zeros((2 * L - 1, 2 * L - 1, nu), dtype=np.complex_)
    for ii in range(nu):
        Bk[:, :, ii] = np.fft.fft2(np.pad(np.reshape(B[:, ii], (L, L)), L // 2))
    Xrec = np.reshape(np.real(B @ z), np.shape(X))
    # %% initial guesses
    gamma_initial = 0.09

    cs = np.zeros((NumGuesses, ne))
    for jj in range(NumGuesses):
        X_initial = np.random.rand(L, L)
        X_initial = X_initial / np.linalg.norm(X_initial)
        _, z_initial, _, _, _ = expand_fb(X_initial, ne)
        c_initial = np.real(T @ z_initial)
        cs[jj, :] = c_initial

    # %% calculations
    y_clean, s, locs = generate_clean_micrograph_2d_rots(c, kvals, Bk, W, L, N, 0.1 * (N / L) ** 2, T, seed=sd)
    psf = full_psf_2d(locs, L)
    tsf = full_tsf_2d(locs, L)

    ExtraMat2, ExtraMat3 = makeExtraMat(L, psf)
    tsfMat = maketsfMat(L, tsf)
    for (idx, SNR) in enumerate(SNRs):
        sigma2 = np.linalg.norm(Xrec) ** 2 / (SNR * np.pi * (L // 2) ** 2)
        y = y_clean + np.random.default_rng().normal(loc=0, scale=np.sqrt(sigma2), size=np.shape(y_clean))

        yy = np.zeros((N, N, 1))
        yy[:, :, 0] = y

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

        for jj in range(NumGuesses):
            X_est_known = Utils.optimization_funcs_rot.optimize_2d_known_psf_triplets(
                np.concatenate((np.reshape(gamma_initial, (1,)), cs[jj, :])), Bk, T, kvals, M1_y, M2_y, M3_y, sigma2, L,
                1, tsfMat, ExtraMat2, ExtraMat3)
            c_est_known = X_est_known.x[1:]
            z_est_known = T.H @ c_est_known
            est_err_coeffs_known = min_err_coeffs(z, z_est_known, kvals)
            errs_known[idx, jj] = est_err_coeffs_known[0]
            costs_known[idx, jj] = X_est_known.fun

            X_est_Algorithm1, _, _ = Utils.optimization_funcs_rot.optimize_rot_Algorithm1_notparallel(
                np.concatenate((np.reshape(gamma_initial, (1,)), cs[jj, :])), Bk, T, kvals, M1_y, M2_y, M3_y, sigma2, L,
                1, W, 10000)
            c_est_Algorithm1 = X_est_Algorithm1.x[1:]
            z_est_Algorithm1 = T.H @ c_est_Algorithm1
            est_err_coeffs_Algorithm1 = min_err_coeffs(z, z_est_Algorithm1, kvals)
            errs_Algorithm1[idx, jj] = est_err_coeffs_Algorithm1[0]
            costs_Algorithm1[idx, jj] = X_est_Algorithm1.fun

    return errs_known, costs_known, errs_Algorithm1, costs_Algorithm1


def calc_err_SNR_comparison(L, ne, N, SNRs, sd):
    """ Calculate estimation error in estimating a specific target image, multiple SNRs.
    For both cases: Algorithm 1 and orcale-deconvolution method.

    Args:
        L: diameter of the target image
        ne: number of expansion coefficients
        N: the size of the micrographs to be generated
        SNRs: an array containing the desired values of SNR
        sd: a seed

    Returns:
        errs_Algorithm1: an array containing the estimation errors for each size, Algorithm 1
        costs_Algorithm1: an array containing the objective function values for each size, Algorithm 1
        errs_conv: an array containing the estimation errors for each size, oracle-based deconvolution
    """
    # %% preliminary definitions
    np.random.seed(sd)
    NumGuesses = 3
    errs_Algorithm1 = np.zeros((len(SNRs), NumGuesses))
    costs_Algorithm1 = np.zeros((len(SNRs), NumGuesses))
    errs_conv = np.zeros((len(SNRs),))
    X = symmetric_target(L, ne)
    X = X / np.linalg.norm(X)

    W = L  # L for arbitrary spacing distribution, 2*L-1 for well-separated

    B, z, roots, kvals, nu = expand_fb(X, ne)
    T = calcT(nu, kvals)
    c = np.real(T @ z)
    z = T.H @ c
    Bk = np.zeros((2 * L - 1, 2 * L - 1, nu), dtype=np.complex_)
    for ii in range(nu):
        Bk[:, :, ii] = np.fft.fft2(np.pad(np.reshape(B[:, ii], (L, L)), L // 2))

    Xrec = np.reshape(np.real(B @ z), np.shape(X))
    Xrec = Xrec / np.linalg.norm(Xrec)
    B, z, roots, kvals, nu = expand_fb(Xrec, ne)
    T = calcT(nu, kvals)
    c = np.real(T @ z)
    z = T.H @ c
    Bk = np.zeros((2 * L - 1, 2 * L - 1, nu), dtype=np.complex_)
    for ii in range(nu):
        Bk[:, :, ii] = np.fft.fft2(np.pad(np.reshape(B[:, ii], (L, L)), L // 2))
    Xrec = np.reshape(np.real(B @ z), np.shape(X))
    # %% initial guesses
    gamma_initial = 0.09

    cs = np.zeros((NumGuesses, ne))
    for jj in range(NumGuesses):
        X_initial = np.random.rand(L, L)
        X_initial = X_initial / np.linalg.norm(X_initial)
        _, z_initial, _, _, _ = expand_fb(X_initial, ne)
        c_initial = np.real(T @ z_initial)
        cs[jj, :] = c_initial

    # %% calculations
    y_clean, s, locs = generate_clean_micrograph_2d_rots(c, kvals, Bk, W, L, N, 0.1 * (N / L) ** 2, T, seed=sd)

    for (idx, SNR) in enumerate(SNRs):
        sigma2 = np.linalg.norm(Xrec) ** 2 / (SNR * np.pi * (L // 2) ** 2)
        y = y_clean + np.random.default_rng().normal(loc=0, scale=np.sqrt(sigma2), size=np.shape(y_clean))

        yy = np.zeros((N, N, 1))
        yy[:, :, 0] = y

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

        for jj in range(NumGuesses):
            X_est_Algorithm1, _, _ = Utils.optimization_funcs_rot. \
                optimize_rot_Algorithm1_notparallel(np.concatenate((np.reshape(gamma_initial, (1,)), cs[jj, :])),
                                                    Bk, T, kvals, M1_y, M2_y, M3_y, sigma2, L, 1, W, 1000)
            c_est_Algorithm1 = X_est_Algorithm1.x[1:]
            z_est_Algorithm1 = T.H @ c_est_Algorithm1
            errs_Algorithm1[idx, jj] = np.linalg.norm(Xrec - np.reshape(np.real(B @ z_est_Algorithm1), np.shape(X))) \
                                       / np.linalg.norm(Xrec)
            costs_Algorithm1[idx, jj] = X_est_Algorithm1.fun
        conv_result = signal.fftconvolve(Xrec, y)

        peaks = photutils.detection.find_peaks(conv_result, threshold=0, box_size=2, npeaks=len(locs))

        x_peaks = (peaks['y_peak']).astype(int)
        y_peaks = (peaks['x_peak']).astype(int)
        peaks_locs = np.zeros((len(x_peaks), 2), dtype=int)
        peaks_locs[:, 0] = x_peaks
        peaks_locs[:, 1] = y_peaks

        X_est_conv = np.zeros((L, L))
        count = 0
        for i in range(len(locs)):
            if peaks_locs[i, 0] >= L - 1 and peaks_locs[i, 1] >= L - 1 and peaks_locs[i, 0] < N and peaks_locs[
                i, 1] < N:
                count += 1
                X_est_conv += y[peaks_locs[i, 0] - L + 1: peaks_locs[i, 0] + 1,
                              peaks_locs[i, 1] - L + 1: peaks_locs[i, 1] + 1]
        X_est_conv = X_est_conv / count
        errs_conv[idx] = np.linalg.norm(Xrec - X_est_conv) / np.linalg.norm(Xrec)

    return errs_Algorithm1, costs_Algorithm1, errs_conv


def calc_err_SNR_comparison_large_target_image(L, ne, N, SNRs, sd):
    """ Calculate estimation error in estimating a specific target image, multiple SNRs.
    For both cases: Algorithm 1 and orcale-deconvolution method.

    Args:
        L: diameter of the target image
        ne: number of expansion coefficients
        N: the size of the micrographs to be generated
        SNRs: an array containing the desired values of SNR
        sd: a seed

    Returns:
        errs_Algorithm1: an array containing the estimation errors for each size, Algorithm 1
        costs_Algorithm1: an array containing the objective function values for each size, Algorithm 1
        errs_conv: an array containing the estimation errors for each size, oracle-based deconvolution
    """
    # %% preliminary definitions
    np.random.seed(sd)
    NumGuesses = 2
    errs_Algorithm1 = np.zeros((len(SNRs), NumGuesses))
    costs_Algorithm1 = np.zeros((len(SNRs), NumGuesses))
    errs_conv = np.zeros((len(SNRs),))
    X = symmetric_target(L, ne)
    X = X / np.linalg.norm(X)

    W = L  # L for arbitrary spacing distribution, 2*L-1 for well-separated

    B, z, roots, kvals, nu = expand_fb(X, ne)
    T = calcT(nu, kvals)
    c = np.real(T @ z)
    z = T.H @ c
    Bk = np.zeros((2 * L - 1, 2 * L - 1, nu), dtype=np.complex_)
    for ii in range(nu):
        Bk[:, :, ii] = np.fft.fft2(np.pad(np.reshape(B[:, ii], (L, L)), L // 2))

    Xrec = np.reshape(np.real(B @ z), np.shape(X))
    Xrec = Xrec / np.linalg.norm(Xrec)
    B, z, roots, kvals, nu = expand_fb(Xrec, ne)
    T = calcT(nu, kvals)
    c = np.real(T @ z)
    z = T.H @ c
    Bk = np.zeros((2 * L - 1, 2 * L - 1, nu), dtype=np.complex_)
    for ii in range(nu):
        Bk[:, :, ii] = np.fft.fft2(np.pad(np.reshape(B[:, ii], (L, L)), L // 2))
    Xrec = np.reshape(np.real(B @ z), np.shape(X))
    # %% initial guesses
    gamma_initial = 0.09

    cs = np.zeros((NumGuesses, ne))
    for jj in range(NumGuesses):
        X_initial = np.random.rand(L, L)
        X_initial = X_initial / np.linalg.norm(X_initial)
        _, z_initial, _, _, _ = expand_fb(X_initial, ne)
        c_initial = np.real(T @ z_initial)
        cs[jj, :] = c_initial

    # %% calculations
    split_axis = 9
    NumMicrographs = split_axis ** 2
    N = (N // split_axis) * split_axis
    y_clean, s, locs = generate_clean_micrograph_2d_rots(c, kvals, Bk, W, L, N, 0.1 * (N / L) ** 2, T, seed=sd)

    for (idx, SNR) in enumerate(SNRs):
        sigma2 = np.linalg.norm(Xrec) ** 2 / (SNR * np.pi * (L // 2) ** 2)
        y = y_clean + np.random.default_rng().normal(loc=0, scale=np.sqrt(sigma2), size=np.shape(y_clean))

        y_blocks = np.split(np.concatenate(np.split(y, split_axis, axis=1)), NumMicrographs)
        start_time = time.time()
        M1_ys, M2_ys, M3_ys = calcM3_parallel_micrographs_given(L, y_blocks, NumMicrographs)
        M1_y = np.mean(M1_ys)

        M2_y = np.mean(M2_ys, 0)

        M3_y = np.mean(M3_ys, 0)
        for jj in range(NumGuesses):
            start_time = time.time()
            X_est_Algorithm1, _, _ = Utils.optimization_funcs_rot. \
                optimize_rot_Algorithm1_notparallel(np.concatenate((np.reshape(gamma_initial, (1,)), cs[jj, :])),
                                                    Bk, T, kvals, M1_y, M2_y, M3_y, sigma2, L, 1, W, 1500, max_iter=250)
            c_est_Algorithm1 = X_est_Algorithm1.x[1:]
            z_est_Algorithm1 = T.H @ c_est_Algorithm1
            errs_Algorithm1[idx, jj] = np.linalg.norm(
                Xrec - np.reshape(np.real(B @ z_est_Algorithm1), np.shape(X))) / np.linalg.norm(Xrec)
            costs_Algorithm1[idx, jj] = X_est_Algorithm1.fun
        conv_result = signal.fftconvolve(Xrec, y)

        peaks = photutils.detection.find_peaks(conv_result, threshold=0, box_size=L // 2, npeaks=len(locs))

        x_peaks = (peaks['y_peak']).astype(int)
        y_peaks = (peaks['x_peak']).astype(int)
        peaks_locs = np.zeros((len(x_peaks), 2), dtype=int)
        peaks_locs[:, 0] = x_peaks
        peaks_locs[:, 1] = y_peaks

        X_est_conv = np.zeros((L, L))
        count = 0
        for i in range(len(locs)):
            if peaks_locs[i, 0] >= L - 1 and peaks_locs[i, 1] >= L - 1 and peaks_locs[i, 0] < N and peaks_locs[
                i, 1] < N:
                count += 1
                X_est_conv += y[peaks_locs[i, 0] - L + 1: peaks_locs[i, 0] + 1,
                              peaks_locs[i, 1] - L + 1: peaks_locs[i, 1] + 1]
        X_est_conv = X_est_conv / count
        errs_conv[idx] = np.linalg.norm(Xrec - X_est_conv) / np.linalg.norm(Xrec)
    return errs_Algorithm1, costs_Algorithm1, errs_conv
