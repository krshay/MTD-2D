# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 14:37:57 2021

@author: Shay Kreymer
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy
from Utils.fb_funcs import expand_fb, calcT
from Utils.generate_clean_micrograph_2d import generate_clean_micrograph_2d_rots
from Utils.funcs_calc_moments import M2_2d, M3_2d
from Utils.psf_tsf_funcs import full_psf_2d, full_tsf_2d, makeExtraMat, maketsfMat
import Utils.optimization_funcs_rot

import shelve

plt.close("all")

np.random.seed(1)
X = np.random.rand(5, 5)
L = np.shape(X)[0]
X = X / np.linalg.norm(X)
W = L # L for arbitrary spacing distribution, 2*L-1 for well-separated

gamma = 0.1
N = 25000
nume = 10
B, z, roots, kvals, nu = expand_fb(X, nume)
T = calcT(nu, kvals)
BT = B @ T.H
c = np.real(T @ z)
z = T.H@c
Xrec = np.reshape(np.real(B @ z), np.shape(X))
Bk = np.zeros((2*L-1, 2*L-1, nu), dtype=np.complex_)
for i in range(nu):
    Bk[ :, :, i] = np.fft.fft2(np.pad(np.reshape(B[ :, i], (L, L)), L//2))

y_clean, s, locs = generate_clean_micrograph_2d_rots(c, kvals, Bk, W, L, N, gamma*(N/L)**2, T, seed=100)

gamma = s[0]*(L/N)**2
SNR = 0.5
sigma2 = 1 / (SNR * L**2)
y = y_clean + np.random.default_rng().normal(loc=0, scale=np.sqrt(sigma2), size=np.shape(y_clean))

kmax = np.max(kvals)

yy = np.zeros((N, N, 1))
yy[ :, :, 0] = y
M1_y = np.mean(y)

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

psf_true = full_psf_2d(locs, L)
tsf_true = full_tsf_2d(locs, L)

ExtraMat2_true, ExtraMat3_true = makeExtraMat(L, psf_true)
tsfMat_true = maketsfMat(L, tsf_true)

X_initial = np.random.rand(L, L)
X_initial = X_initial / np.linalg.norm(X_initial)

_, z_initial, _, _, _ = expand_fb(X_initial, nume)
c_initial = np.real(T @ z_initial)
# %% initiate from gamma = 0.09
gamma_initial_009 = 0.09

y_initial, _, locs_initial = generate_clean_micrograph_2d_rots(c, kvals, Bk, W, L, N, gamma_initial_009*(N/L)**2, T)

# using known PSF and TSF
est_true_009, history_true_009 = Utils.optimization_funcs_rot.optimize_2d_known_psf_triplets_with_callback(np.concatenate((np.reshape(gamma_initial_009, (1,)), c_initial)), Bk, T, kvals, M1_y, M2_y, M3_y, sigma2, L, 1, tsfMat_true, ExtraMat2_true, ExtraMat3_true, numiters=250, gtol=1e-15)
errs_true_009 = np.abs(np.array(history_true_009) - gamma) / gamma

# using approximated PSF and TSF
psf_approx_009 = full_psf_2d(locs_initial, L)
tsf_approx_009 = full_tsf_2d(locs_initial, L)

ExtraMat2_approx_009, ExtraMat3_approx_009 = makeExtraMat(L, psf_approx_009)
tsfMat_approx_009 = maketsfMat(L, tsf_approx_009)

est_approx_009, history_approx_009 = Utils.optimization_funcs_rot.optimize_2d_known_psf_triplets_with_callback(np.concatenate((np.reshape(gamma_initial_009, (1,)), c_initial)), Bk, T, kvals, M1_y, M2_y, M3_y, sigma2, L, 1, tsfMat_approx_009, ExtraMat2_approx_009, ExtraMat3_approx_009, numiters=250, gtol=1e-15)
errs_approx_009 = np.abs(np.array(history_approx_009) - gamma) / gamma

# using no PSF and TSF
ExtraMat2_well_separated_009 = scipy.sparse.csr_matrix(np.zeros(np.shape(ExtraMat2_approx_009)))
ExtraMat3_well_separated_009 = scipy.sparse.csr_matrix(np.zeros(np.shape(ExtraMat3_approx_009)))
tsfMat_well_separated_009 = scipy.sparse.csr_matrix(np.zeros(np.shape(tsfMat_approx_009)))

est_well_separated_009, history_well_separated_009 = Utils.optimization_funcs_rot.optimize_2d_known_psf_triplets_with_callback(np.concatenate((np.reshape(gamma_initial_009, (1,)), c_initial)), Bk, T, kvals, M1_y, M2_y, M3_y, sigma2, L, 1, tsfMat_well_separated_009, ExtraMat2_well_separated_009, ExtraMat3_well_separated_009, numiters=250, gtol=1e-15)
errs_well_separated_009 = 100 * np.abs(np.array(history_well_separated_009) - gamma) / gamma

# %% initiate from gamma = 0.01
gamma_initial_001 = 0.01

y_initial, _, locs_initial = generate_clean_micrograph_2d_rots(c, kvals, Bk, W, L, N, gamma_initial_001*(N/L)**2, T)

# using known PSF and TSF
est_true_001, history_true_001 = Utils.optimization_funcs_rot.optimize_2d_known_psf_triplets_with_callback(np.concatenate((np.reshape(gamma_initial_001, (1,)), c_initial)), Bk, T, kvals, M1_y, M2_y, M3_y, sigma2, L, 1, tsfMat_true, ExtraMat2_true, ExtraMat3_true, numiters=250, gtol=1e-15)
errs_true_001 = np.abs(np.array(history_true_001) - gamma) / gamma

# using approximated PSF and TSF
psf_approx_001_1 = full_psf_2d(locs_initial, L)
tsf_approx_001_1 = full_tsf_2d(locs_initial, L)

ExtraMat2_approx_001_1, ExtraMat3_approx_001_1 = makeExtraMat(L, psf_approx_001_1)
tsfMat_approx_001_1 = maketsfMat(L, tsf_approx_001_1)

est_approx_001_1, history_approx_001_1 = Utils.optimization_funcs_rot.optimize_2d_known_psf_triplets_with_callback(np.concatenate((np.reshape(gamma_initial_001, (1,)), c_initial)), Bk, T, kvals, M1_y, M2_y, M3_y, sigma2, L, 1, tsfMat_approx_001_1, ExtraMat2_approx_001_1, ExtraMat3_approx_001_1, numiters=100, gtol=1e-15)
y_initial, _, locs_2 = generate_clean_micrograph_2d_rots(c, kvals, Bk, W, L, N, history_approx_001_1[-1]*(N/L)**2, T)

psf_approx_001_2 = full_psf_2d(locs_2, L)
tsf_approx_001_2 = full_tsf_2d(locs_2, L)

ExtraMat2_approx_001_2, ExtraMat3_approx_001_2 = makeExtraMat(L, psf_approx_001_2)
tsfMat_approx_001_2 = maketsfMat(L, tsf_approx_001_2)
est_approx_001_2, history_approx_001_2 = Utils.optimization_funcs_rot.optimize_2d_known_psf_triplets_with_callback(np.concatenate((np.reshape(history_approx_001_1[-1], (1,)), c_initial)), Bk, T, kvals, M1_y, M2_y, M3_y, sigma2, L, 1, tsfMat_approx_001_2, ExtraMat2_approx_001_2, ExtraMat3_approx_001_2, numiters=150, gtol=1e-15)
history_approx_001 = history_approx_001_1 + history_approx_001_2[1:]
errs_approx_001_2 = np.abs(np.array(history_approx_001) - gamma) / gamma

# using no PSF and TSF
ExtraMat2_well_separated_001 = scipy.sparse.csr_matrix(np.zeros(np.shape(ExtraMat2_approx_001_1)))
ExtraMat3_well_separated_001 = scipy.sparse.csr_matrix(np.zeros(np.shape(ExtraMat3_approx_001_1)))
tsfMat_well_separated_001 = scipy.sparse.csr_matrix(np.zeros(np.shape(tsfMat_approx_001_1)))

est_well_separated_001, history_well_separated_001 = Utils.optimization_funcs_rot.optimize_2d_known_psf_triplets_with_callback(np.concatenate((np.reshape(gamma_initial_001, (1,)), c_initial)), Bk, T, kvals, M1_y, M2_y, M3_y, sigma2, L, 1, tsfMat_well_separated_001, ExtraMat2_well_separated_001, ExtraMat3_well_separated_001, numiters=250, gtol=1e-15)
errs_well_separated_001 = 100 * np.abs(np.array(history_well_separated_001) - gamma) / gamma



filename=r'C:\Users\kreym\Google Drive\Thesis\Code\MTD-2D-FULL\FINAL_RESULTS\gamma_experiment_data\v2\shelf_experimentb.out'
# %% save
my_shelf = shelve.open(filename,'n') # 'n' for new

for key in dir():
    try:
        my_shelf[key] = globals()[key]
    except TypeError:
        #
        # __builtins__, my_shelf, and imported modules can not be shelved.
        #
        print('ERROR shelving: {0}'.format(key))
    except AttributeError:
        print('ERROR shelving: {0}'.format(key))
my_shelf.close()

# %% load
my_shelf = shelve.open(filename)
for key in my_shelf:
    globals()[key]=my_shelf[key]
my_shelf.close()

# %% plots
with plt.style.context('ieee'):
    plt.close("all")
    iters = list(range(251))
    fig = plt.figure()
    plt.plot(iters, 0.10 * np.ones((251, )), label='_nolegend_')
    l1 = plt.plot(iters, history_true_009)
    l2 = plt.plot(iters, history_approx_009)
    l3 = plt.plot(iters, history_well_separated_009)


    plt.xticks(list(range(0,251,25)))

    plt.xlabel('iterations')

    plt.ylabel('$\gamma$')

    plt.grid(True, which='both', axis='both')

    plt.xlim(0, 250)
    plt.ylim(0.085, 0.11)

    labels = [r"known $\xi$ and $\zeta$", r"approximated $\xi$ and $\zeta$", r"no $\xi$ and $\zeta$"]#, r"accurate $\xi$ and $\zeta$; $\gamma_{\mathrm{init}} = 0.11$", r"approximated $\xi$ and $\zeta$; $\gamma_{\mathrm{init}} = 0.11$"]
    plt.legend(labels, loc=1, fontsize=7)

    fig.tight_layout()
    plt.show()

    plt.savefig(r'C:\Users\kreym\Google Drive\Thesis\Documents\Article v2\figures\gamma_experiment_009.pdf')
    
# %% plots
with plt.style.context('ieee'):
    plt.close("all")
    iters = list(range(251))
    fig = plt.figure()
    plt.plot(iters, 0.10 * np.ones((251, )), label='_nolegend_')
    l1 = plt.plot(iters, history_true_001)
    l2 = plt.plot(iters, history_approx_001)
    l3 = plt.plot(iters, history_well_separated_001)


    plt.xticks(list(range(0,251,25)))
    plt.yticks(list(np.arange(0, 0.12, 0.01)))


    plt.xlabel('iterations')

    plt.ylabel('$\gamma$')

    plt.grid(True, which='both', axis='both')

    plt.xlim(0, 250)
    plt.ylim(0, 0.11)

    labels = [r"known $\xi$ and $\zeta$", r"approximated $\xi$ and $\zeta$", r"no $\xi$ and $\zeta$"]#, r"accurate $\xi$ and $\zeta$; $\gamma_{\mathrm{init}} = 0.11$", r"approximated $\xi$ and $\zeta$; $\gamma_{\mathrm{init}} = 0.11$"]
    plt.legend(labels, loc=4, fontsize=7)

    fig.tight_layout()
    plt.show()

    plt.savefig(r'C:\Users\kreym\Google Drive\Thesis\Documents\Article v2\figures\gamma_experiment_001.pdf')
    
