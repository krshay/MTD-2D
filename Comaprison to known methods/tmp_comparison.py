# -*- coding: utf-8 -*-
"""
Created on Sun Aug 22 16:33:44 2021

@author: Shay Kreymer
"""

# %% imports
import numpy as np
import matplotlib.pyplot as plt

from scipy import signal

import photutils.detection

from Utils.fb_funcs import expand_fb, calcT, min_err_coeffs, symmetric_target
from Utils.generate_clean_micrograph_2d import generate_clean_micrograph_2d_rots
from Utils.funcs_calc_moments import M2_2d, M3_2d
from Utils.psf_tsf_funcs import full_psf_2d, full_tsf_2d, makeExtraMat, maketsfMat
import Utils.optimization_funcs_rot

np.random.seed(1)
L = 5
ne = 10
Xsymm = symmetric_target(L, ne)
X = Xsymm / np.linalg.norm(Xsymm)

W = L # L for arbitrary spacing distribution, 2*L-1 for well-separated

B, z, roots, kvals, nu = expand_fb(X, ne)
T = calcT(nu, kvals)
c = np.real(T @ z)
z = T.H@c
Bk = np.zeros((2*L-1, 2*L-1, nu), dtype=np.complex_)
for ii in range(nu):
    Bk[ :, :, ii] = np.fft.fft2(np.pad(np.reshape(B[ :, ii], (L, L)), L//2))

Xrec = np.reshape(np.real(B @ z), np.shape(X))
gamma = 0.1
N = 7000

gamma_initial = 0.09

X_initial = np.random.rand(L, L)
X_initial = X_initial / np.linalg.norm(X_initial)
_, z_initial, _, _, _ = expand_fb(X_initial, ne)
c_initial = np.real(T @ z_initial)

y_clean, s, locs = generate_clean_micrograph_2d_rots(c, kvals, Bk, W, L, N, gamma*(N/L)**2, T, seed=100)
SNR = 1
sigma2 = 1 / (SNR * L**2)
y = y_clean + np.random.default_rng().normal(loc=0, scale=np.sqrt(sigma2), size=np.shape(y_clean))

psf = full_psf_2d(locs, L)
tsf = full_tsf_2d(locs, L)

ExtraMat2, ExtraMat3 = makeExtraMat(L, psf)
tsfMat = maketsfMat(L, tsf)
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

# X_est_known = Utils.optimization_funcs_rot.optimize_2d_known_psf_triplets(np.concatenate((np.reshape(gamma_initial, (1,)), c_initial)), Bk, T, kvals, M1_y, M2_y, M3_y, sigma2, L, 1, tsfMat, ExtraMat2, ExtraMat3) 
# c_est_known = X_est_known.x[1:]
# z_est_known = T.H @ c_est_known
# est_err_coeffs_known = min_err_coeffs(z, z_est_known, kvals)
# err_known = est_err_coeffs_known[0]
# cost_known = X_est_known.fun

X_est_Algorithm1, _, _ = Utils.optimization_funcs_rot.optimize_rot_Algorithm1_notparallel(np.concatenate((np.reshape(gamma_initial, (1,)), c_initial)), Bk, T, kvals, M1_y, M2_y, M3_y, sigma2, L, 1, W, N) 
c_est_Algorithm1 = X_est_Algorithm1.x[1:]
z_est_Algorithm1 = T.H @ c_est_Algorithm1
est_err_coeffs_Algorithm1 = min_err_coeffs(z, z_est_Algorithm1, kvals)
err_Algorithm1 = est_err_coeffs_Algorithm1[0]
cost_Algorithm1 = X_est_Algorithm1.fun

tmp = signal.fftconvolve(Xrec, y)

peaks = photutils.detection.find_peaks(tmp, threshold=0, box_size=2, npeaks=len(locs))

x_peaks = (peaks['y_peak']).astype(int)
y_peaks = (peaks['x_peak']).astype(int)
peaks_locs = np.zeros((len(x_peaks), 2), dtype=int)
peaks_locs[ :, 0] = x_peaks
peaks_locs[ :, 1] = y_peaks

X_est_conv = np.zeros((L, L))
for i in range(len(locs)):
    X_est_conv += y[peaks_locs[i, 0] - L + 1: peaks_locs[i, 0] + 1,  peaks_locs[i, 1] - L + 1: peaks_locs[i, 1] + 1]
X_est_conv = X_est_conv / len(locs)
_, z_est_conv, _, _, _ = expand_fb(X, ne)
err_conv = min_err_coeffs(z, z_est_conv, kvals)[0]

