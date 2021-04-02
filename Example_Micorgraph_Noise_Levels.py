# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 20:26:09 2020

@author: kreym
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

import random
import string
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
random.seed(100)
if __name__ == '__main__':
    
    # script_dir = os.path.dirname(__file__)
    # rel_path = "images/molecule17.png"
    # file_path = os.path.join(script_dir, rel_path)
    X = plt.imread("images/molecule9.png")
    # X = phantominator.ct_shepp_logan(17)
    # X = np.load('X_data.npy')
    L = np.shape(X)[0]
    X = L**2 * X / np.linalg.norm(X)
    W = L # L for arbitrary spacing distribution, 2*L-1 for well-separated

    N = 70
    ne = 50
    B, z, roots, kvals, nu = expand_fb(X, ne)
    T = calcT(nu, kvals)
    BT = B @ T.H
    c = np.real(T @ z)
    z = T.H@c
    Xrec = np.reshape(np.real(B @ z), np.shape(X))
    XrecT = np.reshape(np.real(BT @ c), np.shape(X))
    theta = np.random.uniform(0, 2*np.pi)
    Bk = np.zeros((2*L-1, 2*L-1, nu), dtype=np.complex_)
    for i in range(nu):
        Bk[ :, :, i] = np.fft.fft2(np.pad(np.reshape(B[ :, i], (L, L)), L//2))
    
    Xrot_freq = rot_img_freq(theta, z, kvals, Bk, L)
    Xrot_freqT = rot_img_freqT(theta, c, kvals, Bk, L, T)
    y_clean, s, locs = generate_clean_micrograph_2d_rots(c, kvals, Bk, W, L, N, 0.30*(N/L)**2, T)
    # y_clean_triplets, s_triplets, locs_triplets = generate_clean_micrograph_2d_rots(c, kvals, Bk, W, L, N, 0.10*(N/L)**2, T)
    
    gamma = s[0]*(L/N)**2
    # gamma_tripltes = s_triplets[0]*(L/N)**2
    SNR = 0.1
    sigma2 = np.linalg.norm(Xrec)**2 / (SNR * np.pi * (L//2)**2)
    y1 = y_clean + np.random.default_rng().normal(loc=0, scale=np.sqrt(sigma2), size=np.shape(y_clean))
    
    SNR = 0
    sigma2 = 0
    y2 = y_clean + np.random.default_rng().normal(loc=0, scale=np.sqrt(sigma2), size=np.shape(y_clean))
    
    SNR = 10
    sigma2 = np.linalg.norm(Xrec)**2 / (SNR * np.pi * (L//2)**2)
    y3 = y_clean + np.random.default_rng().normal(loc=0, scale=np.sqrt(sigma2), size=np.shape(y_clean))
    
    width = 3.487
    height = width / 1.618
    plt.close("all")
    ttls = ['SNR = 10', 'SNR = 0.1']
    ys = [y3, y1, y2]
    for n in range(3):
        fig = plt.figure()#subplots(1, 2)
    
        ax = plt.axes()
        im = ax.imshow(ys[n], cmap="gray")   
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = plt.colorbar(im, cax=cax)
        cbar.ax.tick_params(labelsize=20)
        # ax.text(-0.1, 1.03, '('+string.ascii_lowercase[n]+')', transform=ax.transAxes, 
                # size=30, weight='bold')
        ax.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
        # ax.title.set_text(ttls[n])
        # ax.title.set_size(30)
        fig.set_size_inches(4*width, 4*height)
        if n == 0:
            fig.savefig(r'C:\Users\kreym\Google Drive\Thesis\Documents\Article\figures\Micrographs_noise_a.pdf', bbox_inches='tight')
        else:
            if n == 1:
                fig.savefig(r'C:\Users\kreym\Google Drive\Thesis\Documents\Article\figures\Micrographs_noise_b.pdf', bbox_inches='tight')
            else:
                fig.savefig(r'C:\Users\kreym\Google Drive\Thesis\Documents\Article\figures\Micrographs_noise_clean.pdf', bbox_inches='tight')


        