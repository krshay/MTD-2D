# -*- coding: utf-8 -*-
"""
Created on Sat Jan 16 11:00:07 2021

@author: kreym
"""

import numpy as np
import matplotlib.pyplot as plt
from Utils.mpl_toolkits.axes_grid1 import make_axes_locatable

import random

from Utils.PIL import Image

from Utils.fb_funcs import expand_fb, rot_img, min_err_rots, min_err_coeffs, rot_img_freq, calcT, rot_img_freqT
from Utils.generate_clean_micrograph_2d import generate_clean_micrograph_2d_one_neighbor_rots, generate_clean_micrograph_2d_rots

random.seed(150)

plt.close("all")

if __name__ == '__main__':

    X = plt.imread("../images/tiger65.png")

    L = np.shape(X)[0]
    X = L**2 * X / np.linalg.norm(X)
    W = L # L for arbitrary spacing distribution, 2*L-1 for well-separated

    N = 200
    ne = 2000
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
    y_clean, s, locs = generate_clean_micrograph_2d_rots(c, kvals, Bk, W, L, N, 3, T)
    # y_clean_triplets, s_triplets, locs_triplets = generate_clean_micrograph_2d_rots(c, kvals, Bk, W, L, N, 0.10*(N/L)**2, T)
    yy_clean, s, locs = generate_clean_micrograph_2d_rots(c, kvals, Bk, 2*L-1, L, N, 3, T)

    gamma = s[0]*(L/N)**2
    # gamma_tripltes = s_triplets[0]*(L/N)**2
    sigma2 = 0
    y = y_clean + np.random.default_rng().normal(loc=0, scale=np.sqrt(sigma2), size=np.shape(y_clean))
    yy = yy_clean + np.random.default_rng().normal(loc=0, scale=np.sqrt(sigma2), size=np.shape(y_clean))
    y1 = np.pad(y, ((0, 20), (0, 35)))
    
    fig = plt.figure()

    ax = plt.axes()
    im = ax.imshow(y1, cmap="gray")   

    ax.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)

    fig.savefig(r'C:\Users\kreym\Google Drive\Thesis\Documents\Article\Micrograph_neighbors_triplets_1.png', bbox_inches='tight')
    Image1 = Image.open(r'C:\Users\kreym\Google Drive\Thesis\Documents\Article\Micrograph_neighbors_triplets_1.png')
    yy1 = np.pad(yy, ((0, 20), (0, 35)))
    
    fig = plt.figure()

    ax = plt.axes()
    im = ax.imshow(yy1, cmap="gray")   

    ax.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)

    fig.savefig(r'C:\Users\kreym\Google Drive\Thesis\Documents\Article\Micrograph_neighbors_triplets_11.png', bbox_inches='tight')
    Image11 = Image.open(r'C:\Users\kreym\Google Drive\Thesis\Documents\Article\Micrograph_neighbors_triplets_11.png')

    y2 = np.pad(y, ((15, 5), (35, 0)))
    
    fig = plt.figure()

    ax = plt.axes()
    im = ax.imshow(y2, cmap="gray")   

    ax.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)

    fig.savefig(r'C:\Users\kreym\Google Drive\Thesis\Documents\Article\Micrograph_neighbors_triplets_2.png', bbox_inches='tight')
    Image2 = Image.open(r'C:\Users\kreym\Google Drive\Thesis\Documents\Article\Micrograph_neighbors_triplets_2.png')

    yy2 = np.pad(yy, ((15, 5), (35, 0)))
    
    fig = plt.figure()

    ax = plt.axes()
    im = ax.imshow(yy2, cmap="gray")   

    ax.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)

    fig.savefig(r'C:\Users\kreym\Google Drive\Thesis\Documents\Article\Micrograph_neighbors_triplets_22.png', bbox_inches='tight')
    Image22 = Image.open(r'C:\Users\kreym\Google Drive\Thesis\Documents\Article\Micrograph_neighbors_triplets_22.png')

    y3 = np.pad(y, ((20, 0,), (2, 33)))
    
    fig = plt.figure()

    ax = plt.axes()
    im = ax.imshow(y3, cmap="gray")   

    ax.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)

    fig.savefig(r'C:\Users\kreym\Google Drive\Thesis\Documents\Article\Micrograph_neighbors_triplets_3.png', bbox_inches='tight')
    Image3 = Image.open(r'C:\Users\kreym\Google Drive\Thesis\Documents\Article\Micrograph_neighbors_triplets_3.png')

    yy3 = np.pad(yy, ((20, 0,), (2, 33)))
    
    fig = plt.figure()

    ax = plt.axes()
    im = ax.imshow(yy3, cmap="gray")   

    ax.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)

    fig.savefig(r'C:\Users\kreym\Google Drive\Thesis\Documents\Article\Micrograph_neighbors_triplets_33.png', bbox_inches='tight')
    Image33 = Image.open(r'C:\Users\kreym\Google Drive\Thesis\Documents\Article\Micrograph_neighbors_triplets_33.png')

    Img = Image.blend(Image.blend(Image1, Image2, 0.5), Image3, 1/3)
    fig = plt.figure()
    ax = plt.axes()
    im = ax.imshow(Img, cmap="gray")   

    ax.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
    fig.savefig(r'C:\Users\kreym\Google Drive\Thesis\Documents\Article\Micrograph_neighbors_triplets_clean.pdf', bbox_inches='tight')

    Imgg = Image.blend(Image.blend(Image11, Image22, 0.5), Image33, 1/3)
    fig = plt.figure()
    ax = plt.axes()
    im = ax.imshow(Imgg, cmap="gray")

    ax.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
    fig.savefig(r'C:\Users\kreym\Google Drive\Thesis\Documents\Article\Micrograph_neighbors_triplets_clean_well_separated.pdf', bbox_inches='tight')

        