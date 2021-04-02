# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 14:07:04 2021

@author: kreym
"""

# %% imports
import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.axes_grid1 import make_axes_locatable

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


X = plt.imread("images/molecule9.png")
L = np.shape(X)[0]
W = L # L for arbitrary spacing distribution, 2*L-1 for well-separated

ne = 40
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
cbar = plt.colorbar(im, cax=cax)
cbar.ax.tick_params(labelsize=20)
# ax.text(-0.1, 1.03, '('+string.ascii_lowercase[n]+')', transform=ax.transAxes, 
        # size=30, weight='bold')
ax.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
fig.tight_layout()
plt.show()
    
plt.savefig(r'C:\Users\kreym\Google Drive\Thesis\Documents\Article\figures\recovery_true.pdf')


    