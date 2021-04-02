# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 14:37:57 2021

@author: kreym
"""
# %% imports
import numpy as np
import matplotlib.pyplot as plt

import scipy

from fb_funcs import expand_fb, calcT
from generate_clean_micrograph_2d import generate_clean_micrograph_2d_rots
from funcs_calc_moments import M2_2d, M3_2d
from psf_functions_2d import full_psf_2d
from tsf_functions_2d import full_tsf_2d
import optimization_funcs_rot
from makeExtraMat import makeExtraMat
from maketsfMat import maketsfMat

# %% main
plt.close("all")

np.random.seed(100)
X = np.random.rand(5, 5)
L = np.shape(X)[0]
X = X / np.linalg.norm(X)
W = L # L for arbitrary spacing distribution, 2*L-1 for well-separated

gamma = 0.10
N = 25000
ne = 10
B, z, roots, kvals, nu = expand_fb(X, ne)
T = calcT(nu, kvals)
BT = B @ T.H
c = np.real(T @ z)
z = T.H@c
Xrec = np.reshape(np.real(B @ z), np.shape(X))
Bk = np.zeros((2*L-1, 2*L-1, nu), dtype=np.complex_)
for i in range(nu):
    Bk[ :, :, i] = np.fft.fft2(np.pad(np.reshape(B[ :, i], (L, L)), L//2))

y_clean, s, locs = generate_clean_micrograph_2d_rots(c, kvals, Bk, W, L, N, gamma*(N/L)**2, T)

gamma = s[0]*(L/N)**2
sigma2 = 0.1
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
    print(i1)
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

_, z_initial, _, _, _ = expand_fb(X_initial, ne)
c_initial = np.real(T @ z_initial)
# %% initiate from 0.09
gamma_initial_009 = 0.090

y_initial, _, locs_initial = generate_clean_micrograph_2d_rots(c, kvals, Bk, W, L, N, gamma_initial_009*(N/L)**2, T)

psf_approx_009 = full_psf_2d(locs_initial, L)
tsf_approx_009 = full_tsf_2d(locs_initial, L)

ExtraMat2_approx_009, ExtraMat3_approx_009 = makeExtraMat(L, psf_approx_009)
tsfMat_approx_009 = maketsfMat(L, tsf_approx_009)

ExtraMat2_well_separated_009 = scipy.sparse.csr_matrix(np.zeros(np.shape(ExtraMat2_approx_009)))
ExtraMat3_well_separated_009 = scipy.sparse.csr_matrix(np.zeros(np.shape(ExtraMat3_approx_009)))
tsfMat_well_separated_009 = scipy.sparse.csr_matrix(np.zeros(np.shape(tsfMat_approx_009)))

est_true_009, history_true_009 = optimization_funcs_rot.optimize_2d_known_psf_triplets_with_callback(np.concatenate((np.reshape(gamma_initial_009, (1,)), c_initial)), Bk, T, kvals, M1_y, M2_y, M3_y, sigma2, L, 1, tsfMat_true, ExtraMat2_true, ExtraMat3_true, numiters=100, gtol=1e-15)
errs_true_009 = np.abs(np.array(history_true_009) - gamma) / gamma

est_approx_009, history_approx_009 = optimization_funcs_rot.optimize_2d_known_psf_triplets_with_callback(np.concatenate((np.reshape(gamma_initial_009, (1,)), c_initial)), Bk, T, kvals, M1_y, M2_y, M3_y, sigma2, L, 1, tsfMat_approx_009, ExtraMat2_approx_009, ExtraMat3_approx_009, numiters=100, gtol=1e-15)
errs_approx_009 = np.abs(np.array(history_approx_009) - gamma) / gamma

est_well_separated_009, history_well_separated_009 = optimization_funcs_rot.optimize_2d_known_psf_triplets_with_callback(np.concatenate((np.reshape(gamma_initial_009, (1,)), c_initial)), Bk, T, kvals, M1_y, M2_y, M3_y, sigma2, L, 1, tsfMat_well_separated_009, ExtraMat2_well_separated_009, ExtraMat3_well_separated_009, numiters=100, gtol=1e-15)
errs_well_separated_009 = 100 * np.abs(np.array(history_well_separated_009) - gamma) / gamma

# %% initiate from 0.11
gamma_initial_011 = 0.11

y_initial, _, locs_initial = generate_clean_micrograph_2d_rots(c, kvals, Bk, W, L, N, gamma_initial_011*(N/L)**2, T)

psf_approx_011 = full_psf_2d(locs_initial, L)
tsf_approx_011 = full_tsf_2d(locs_initial, L)

ExtraMat2_approx_011, ExtraMat3_approx_011 = makeExtraMat(L, psf_approx_011)
tsfMat_approx_011 = maketsfMat(L, tsf_approx_011)

est_true_011, history_true_011 = optimization_funcs_rot.optimize_2d_known_psf_triplets_with_callback(np.concatenate((np.reshape(gamma_initial_011, (1,)), c_initial)), Bk, T, kvals, M1_y, M2_y, M3_y, sigma2, L, 1, tsfMat_true, ExtraMat2_true, ExtraMat3_true, numiters=100, gtol=1e-15)
errs_true_011 = np.abs(np.array(history_true_011) - gamma) / gamma

est_approx_011, history_approx_011 = optimization_funcs_rot.optimize_2d_known_psf_triplets_with_callback(np.concatenate((np.reshape(gamma_initial_011, (1,)), c_initial)), Bk, T, kvals, M1_y, M2_y, M3_y, sigma2, L, 1, tsfMat_approx_011, ExtraMat2_approx_011, ExtraMat3_approx_011, numiters=100, gtol=1e-15)
errs_approx_011 = np.abs(np.array(history_approx_011) - gamma) / gamma

# %% plots
with plt.style.context('ieee'):
    plt.close("all")
    iters = list(range(101))
    fig = plt.figure()
    plt.plot(iters, 0.10 * np.ones((101, )), label='_nolegend_')
    l1 = plt.plot(iters, history_true_009)
    l2 = plt.plot(iters, history_approx_009)
    l3 = plt.plot(iters, history_well_separated_009)
    # l3 = ax[0].plot(iters, history_true_011)
    # l4 = ax[0].plot(iters, history_approx_011)
    
    # ax[1].plot(iters, 1000 * np.ones((101, )))
    # ax[1].plot(iters, errs_true_009)
    # ax[1].plot(iters, errs_approx_009)
    # ax[1].plot(iters, errs_well_separated_009)
    # ax[1].plot(iters, errs_true_011)
    # ax[1].plot(iters, errs_approx_011)

    plt.xticks(list(range(0,101,10)))
    # ax[0].get_shared_x_axes().join(ax[0], ax[1])
    # ax[0].set_xticks(list(range(0,101,10)))
    # ax[0].set_xticklabels([])
    
    plt.xlabel('iterations')
    
    plt.ylabel('$\gamma$')
    
    # fig.set_ylabel('error')
    
    plt.grid(True, which='both', axis='both')
    
    plt.xlim(0, 100)
    plt.ylim(0.09, 0.11)
    # ax[1].set_xlim(0, 100)
    # ax[1].set_ylim(0, 0.10)
    
    # ax[1].grid(True, which='both', axis='both')
    
    labels = [r"known $\xi$ and $\zeta$", r"approximated $\xi$ and $\zeta$", r"no $\xi$ and $\zeta$"]#, r"accurate $\xi$ and $\zeta$; $\gamma_{\mathrm{init}} = 0.11$", r"approximated $\xi$ and $\zeta$; $\gamma_{\mathrm{init}} = 0.11$"]
    plt.legend(labels, loc=1, fontsize=7)

    # plt.figlegend([l1, l2, l3, l4], labels=line_labels, bbox_to_anchor = (0,-0.1,1,1), bbox_transform = plt.gcf().transFigure)
    fig.tight_layout()
    plt.show()
    
    plt.savefig(r'C:\Users\kreym\Google Drive\Thesis\Documents\Article\figures\gamma_experiment.pdf')


# %% legend
colors = ["b", "k", "r", "g"]
styles = ['-', '--', '-.', ':']
f = lambda m,c,st: plt.plot([],[],marker=m, color=c, ls=st)[0]
handles = [f(" ", colors[i], styles[i]) for i in range(4)]
labels = [r"known $\xi$ and $\zeta$; $\gamma_{\mathrm{init}} = 0.09$", r"approximated $\xi$ and $\zeta$; $\gamma_{\mathrm{init}} = 0.09$"]#, r"accurate $\xi$ and $\zeta$; $\gamma_{\mathrm{init}} = 0.11$", r"approximated $\xi$ and $\zeta$; $\gamma_{\mathrm{init}} = 0.11$"]

legend = plt.legend(handles, labels, loc=3, framealpha=1, frameon=True)

def export_legend(legend, filename=r'C:\Users\kreym\Google Drive\Thesis\Documents\Article\figures\gamma_experiment_legend.pdf', expand=[-5,-5,5,5]):
    fig  = legend.figure
    fig.canvas.draw()
    bbox  = legend.get_window_extent()
    bbox = bbox.from_extents(*(bbox.extents + np.array(expand)))
    bbox = bbox.transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(filename, dpi="figure", bbox_inches=bbox)

export_legend(legend)
plt.show()

# # %% saves
# np.save('Results/gamma_exp/iters', np.array(iters))
# np.save('Results/gamma_exp/history_true_009', np.array(history_true_009))
# np.save('Results/gamma_exp/history_approx_009', np.array(history_approx_009))    
# np.save('Results/gamma_exp/history_true_011', np.array(history_true_011))    
# np.save('Results/gamma_exp/history_approx_011', np.array(history_approx_011))    
# np.save('Results/gamma_exp/errs_true_009', errs_true_009)
# np.save('Results/gamma_exp/errs_approx_009', errs_approx_009)    
# np.save('Results/gamma_exp/errs_true_011', errs_true_011)    
# np.save('Results/gamma_exp/errs_approx_011', errs_approx_011)
# np.save('Results/gamma_exp/history_well_separated_009', np.array(history_well_separated_009))
# np.save('Results/gamma_exp/errs_well_separated_009', errs_well_separated_009)
# %% loads
errs_approx_009 = np.load('Results/gamma_exp/errs_approx_009.npy')
errs_approx_009 = errs_approx_009 / 100
errs_approx_011 = np.load('Results/gamma_exp/errs_approx_011.npy')
errs_true_009 = np.load('Results/gamma_exp/errs_true_009.npy')
errs_true_009 = errs_true_009 / 100
errs_true_011 = np.load('Results/gamma_exp/errs_true_011.npy')
errs_well_separated_009 = np.load('Results/gamma_exp/errs_well_separated_009.npy')
errs_well_separated_009 = errs_well_separated_009 / 100
history_approx_009 = np.load('Results/gamma_exp/history_approx_009.npy')
history_approx_011 = np.load('Results/gamma_exp/history_approx_011.npy')
history_true_009 = np.load('Results/gamma_exp/history_true_009.npy')
history_true_011 = np.load('Results/gamma_exp/history_true_011.npy')
history_well_separated_009 = np.load('Results/gamma_exp/history_well_separated_009.npy')


    