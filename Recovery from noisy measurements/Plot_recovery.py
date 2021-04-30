# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 14:07:04 2021

@author: Shay Kreymer
"""

# %% imports
import numpy as np
import matplotlib.pyplot as plt

from Utils.fb_funcs import expand_fb, calcT


# %% main
plt.close("all")

X = plt.imread("../images/molecule9.png")
X = X * 10
L = np.shape(X)[0]
W = L # L for arbitrary spacing distribution, 2*L-1 for well-separated

ne = 34
B, z, roots, kvals, nu = expand_fb(X, ne)
T = calcT(nu, kvals)
BT = B @ T.H
c = np.real(T @ z)
z = T.H@c
Xrec_true = np.reshape(np.real(B @ z), np.shape(X))

# %% plots
with plt.style.context('ieee'):
    # %% Original
    fig = plt.figure()
    
    ax = plt.axes()
    im = ax.imshow(Xrec_true)

    ax.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
    fig.tight_layout()
    plt.show()
        
    plt.savefig(r'C:\Users\kreym\Google Drive\Thesis\Documents\Article\figures\recovery_true.pdf')
    
    # %% SNR = 10
    z_est_best_SNR10 = np.load("../Results/Recovery/z_est_best_SNR_10.npy")
    X_SNR10 = np.reshape(np.real(B @ z_est_best_SNR10), np.shape(X))
    
    fig = plt.figure()
    
    ax = plt.axes()
    im = ax.imshow(X_SNR10)

    ax.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
    fig.tight_layout()
    plt.show()
        
    plt.savefig(r'C:\Users\kreym\Google Drive\Thesis\Documents\Article\figures\recovery_SNR_10.pdf')
    
    # %% SNR = 0.5
    z_est_best_SNR_05 = np.load("../Results/Recovery/z_est_best_SNR_05.npy")
    X_SNR_05 = np.reshape(np.real(B @ z_est_best_SNR_05), np.shape(X))
    
    fig = plt.figure()
    
    ax = plt.axes()
    im = ax.imshow(X_SNR_05)

    ax.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
    fig.tight_layout()
    plt.show()
        
    plt.savefig(r'C:\Users\kreym\Google Drive\Thesis\Documents\Article\figures\recovery_SNR_05.pdf')
    