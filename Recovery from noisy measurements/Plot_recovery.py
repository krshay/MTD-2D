# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 14:07:04 2021

@author: Shay Kreymer
"""

# %% imports
import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.axes_grid1 import make_axes_locatable

import scipy

from Utils.fb_funcs import expand_fb, calcT
from Utils.funcs_calc_moments import M2_2d, M3_2d
from Utils.psf_tsf_funcs import full_psf_2d
from Utils.psf_tsf_funcs import full_tsf_2d
import Utils.optimization_funcs_rot
from Utils.psf_tsf_funcs import makeExtraMat
from Utils.psf_tsf_funcs import maketsfMat

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
# with plt.style.context('ieee'):
fig = plt.figure()

ax = plt.axes()
im = ax.imshow(Xrec_true)
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
im.set_clim(0, 7)
cbar = plt.colorbar(im, cax=cax)
cbar.ax.tick_params(labelsize=20)
# ax.text(-0.1, 1.03, '('+string.ascii_lowercase[n]+')', transform=ax.transAxes, 
        # size=30, weight='bold')
ax.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
fig.tight_layout()
plt.show()
    
plt.savefig(r'C:\Users\kreym\Google Drive\Thesis\Documents\Article\figures\recovery_true.pdf')



z_est_best_no_noise = np.load("../Results/Recovery/z_est_best_no_noise.npy")
X_no_noise = np.reshape(np.real(B @ z_est_best_no_noise), np.shape(X))

fig = plt.figure()

ax = plt.axes()
im = ax.imshow(X_no_noise)
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
im.set_clim(0, 7)
cbar = plt.colorbar(im, cax=cax)
cbar.ax.tick_params(labelsize=20)
# ax.text(-0.1, 1.03, '('+string.ascii_lowercase[n]+')', transform=ax.transAxes, 
        # size=30, weight='bold')
ax.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
fig.tight_layout()
plt.show()
    
plt.savefig(r'C:\Users\kreym\Google Drive\Thesis\Documents\Article\figures\recovery_no_noise.pdf')


z_est_best_SNR10 = np.load("../Results/Recovery/z_est_best_SNR10.npy")
X_SNR10 = np.reshape(np.real(B @ z_est_best_SNR10), np.shape(X))

fig = plt.figure()

ax = plt.axes()
im = ax.imshow(X_SNR10)
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
im.set_clim(0, 7)
cbar = plt.colorbar(im, cax=cax)
cbar.ax.tick_params(labelsize=20)
# ax.text(-0.1, 1.03, '('+string.ascii_lowercase[n]+')', transform=ax.transAxes, 
        # size=30, weight='bold')
ax.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
fig.tight_layout()
plt.show()
    
plt.savefig(r'C:\Users\kreym\Google Drive\Thesis\Documents\Article\figures\recovery_SNR10.pdf')

    