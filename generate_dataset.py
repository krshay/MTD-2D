# -*- coding: utf-8 -*-
"""
Created on Sat Aug 22 19:49:39 2020

@author: kreym
"""

import numpy as np

from fb_funcs import expand_fb
from funcs_calc_moments_rot import calck1k2k3, calcS3_x_grad, calcS3_x_neigh_grad
from generate_signal import generate_signal

# def calcS3(shift1, shift2, S3_x, gS3_x, S3_x_neigh, gS3_x_neigh, L, psf):
#     shift1y = shift1[0]
#     shift1x = shift1[1]
    
#     shift2y = shift2[0]
#     shift2x = shift2[1]
            
#     # %% 1.
#     M3_clean = S3_x[shift1y, shift1x, shift2y, shift2x]
#     T3_clean = gS3_x[shift1y, shift1x, shift2y, shift2x]
    
#     # %% 2. ->
#     M3k_extra = 0
#     T3k_extra = np.zeros_like(T3_clean)
    
#     # %% 2. 
#     for j in range(max(shift1y, shift2y) - (L-1), L + min(shift1y, shift2y)):
#         for i in range(max(shift1x, shift2x) - (L-1), L + min(shift1x, shift2x)):
#             if not (np.abs(i) < L and np.abs(j) < L):
#                 M3k_extra = M3k_extra + evaluate_psf_full(psf, (j,i), L) * S3_x_neigh[shift2y-shift1y, shift2x-shift1x, j-shift1y, i-shift1x]
#                 T3k_extra = T3k_extra + evaluate_psf_full(psf, (j,i), L) * gS3_x_neigh[shift2y-shift1y, shift2x-shift1x, j-shift1y, i-shift1x]
           
#     # %% 3. 
#     for j in range(shift1y - (L-1), L + shift1y - shift2y):
#         for i in range(shift1x - (L-1), L + shift1x - shift2x):
#             if not (np.abs(i) < L and np.abs(j) < L):
#                 M3k_extra = M3k_extra + evaluate_psf_full(psf, (j,i), L) * S3_x_neigh[j-shift1y, i-shift1x, j+shift2y-shift1y, i+shift2x-shift1x]
#                 T3k_extra = T3k_extra + evaluate_psf_full(psf, (j,i), L) * gS3_x_neigh[j-shift1y, i-shift1x, j+shift2y-shift1y, i+shift2x-shift1x]
    
#     # %% 4. 
#     for j in range(shift2y - (L-1), L + shift2y - shift1y):
#         for i in range(shift2x - (L-1), L + shift2x - shift1x):
#             if not (np.abs(i) < L and np.abs(j) < L):
#                 M3k_extra = M3k_extra + evaluate_psf_full(psf, (j,i), L) * S3_x_neigh[j-shift2y, i-shift2x, j+shift1y-shift2y, i+shift1x-shift2x]
#                 T3k_extra = T3k_extra + evaluate_psf_full(psf, (j,i), L) * gS3_x_neigh[j-shift2y, i-shift2x, j+shift1y-shift2y, i+shift1x-shift2x]
            
#     return M3_clean, T3_clean, M3k_extra, T3k_extra

# L = 5

# X = np.random.rand(L, L)

# N = 2000
# ne = 5
# B, z, roots, kvals, nu = expand_fb(X, ne)
# Bk = np.zeros((2*L-1, 2*L-1, ne), dtype=np.complex_)
# for i in range(ne):
#     Bk[ :, :, i] = np.fft.fft2(np.pad(np.reshape(B[ :, i], (L, L)), L//2))
    

# y_init, s_init, locs_init = generate_clean_micrograph_2d_one_neighbor_rots(z, kvals, B, L, N, 0.10*(N/L)**2)

# psf_init = full_psf_2d(locs_init, L)

# k1k2k3_map = calck1k2k3(L)
# NumIters = 1000000
# acs = np.zeros((NumIters, L**4 + 1), dtype=np.complex_)
# grds = np.zeros((NumIters, (L**4)*ne), dtype=np.complex_)

# acs_neigh = np.zeros((NumIters, L**4), dtype=np.complex_)
# grds_neigh = np.zeros((NumIters, (L**4)*ne), dtype=np.complex_)

# for i in range(NumIters):
#     print(i)
#     if i % 1000 == 0:
#         np.save(f'acs_{i}.npy', acs[:i, :])
#         np.save(f'grds_{i}.npy', grds[:i, :])
#         np.save(f'acs_neigh_{i}.npy', acs_neigh[:i, :])
#         np.save(f'grds_neigh_{i}.npy', grds_neigh[:i, :])
#     X = np.random.rand(L, L)
#     B, z, roots, kvals, nu = expand_fb(X, ne)
#     gamma = random.uniform(0, 0.20)
#     acs[i, 0] = gamma
#     y_init, s_init, locs_init = generate_clean_micrograph_2d_one_neighbor_rots(z, kvals, B, L, N, gamma*(N/L)**2)
#     psf_init = full_psf_2d(locs_init, L)
    
    
#     kmax = np.max(kvals)
#     Nmax = 6*kmax
#     S3_x, gS3_x = calcS3_x_grad(L, Nmax, Bk, z, kvals, k1k2k3_map)
    
#     Nmax_neigh = 4*kmax
#     S3_x_neigh, gS3_x_neigh = calcS3_x_neigh_grad(L, Nmax_neigh, Bk, z, kvals, k1k2k3_map)
    
#     S3_clean = np.zeros((L, L, L, L), dtype=np.complex_)
#     gS3_clean = np.zeros((L, L, L, L, len(z)), dtype=np.complex_)
#     S3_extra = np.zeros((L, L, L, L), dtype=np.complex_)
#     gS3_extra = np.zeros((L, L, L, L, len(z)), dtype=np.complex_)
#     for i1 in range(L):
#         for j1 in range(L):
#             for i2 in range(L):
#                 for j2 in range(L):
#                     S3_clean[i1, j1, i2, j2], gS3_clean[i1, j1, i2, j2, :], S3_extra[i1, j1, i2, j2], gS3_extra[i1, j1, i2, j2, :] = calcS3((j1, i1), (j2, i2), S3_x, gS3_x, S3_x_neigh, gS3_x_neigh, L, psf_init)
                    
#     acs[i, 1:] = S3_clean.flatten()
#     grds[i, :] = gS3_clean.flatten()
#     acs_neigh[i, :] = S3_extra.flatten()
#     grds_neigh[i, :] = gS3_extra.flatten()


L = 5

ne = 5
X = np.random.rand(L, L)
B, z, roots, kvals, nu = expand_fb(X, ne)
Bk = np.zeros((2*L-1, 2*L-1, ne), dtype=np.complex_)
for i in range(ne):
    Bk[ :, :, i] = np.fft.fft2(np.pad(np.reshape(B[ :, i], (L, L)), L//2))
    
kmax = np.max(kvals)
k1k2k3_map = calck1k2k3(L)
NumIters = 10

acs = np.zeros((NumIters, (2*L-1)**4))
grds = np.zeros((NumIters, ((2*L-1)**4)*ne), dtype=np.complex_)

acs_neigh = np.zeros((NumIters, (2*L-1)**4))
grds_neigh = np.zeros((NumIters, ((2*L-1)**4)*ne), dtype=np.complex_)

for i in range(NumIters):
    X = np.random.rand(L, L)
    X = L*L * X / np.linalg.norm(X)
    _, z, _, _, _ = expand_fb(X, ne)
    print(i)
    if i % 1000 == 0:
        np.save(f'acs_{i}.npy', acs[:i, :])
        np.save(f'grds_{i}.npy', grds[:i, :])
        np.save(f'acs_neigh_{i}.npy', acs_neigh[:i, :])
        np.save(f'grds_neigh_{i}.npy', grds_neigh[:i, :])


    Nmax = 6*kmax
    S3_x, gS3_x = calcS3_x_grad(L, Nmax, Bk, z, kvals, k1k2k3_map)
    
    Nmax_neigh = 4*kmax
    S3_x_neigh, gS3_x_neigh = calcS3_x_neigh_grad(L, Nmax_neigh, Bk, z, kvals, k1k2k3_map)
    
    
    acs[i, :] = np.real(S3_x).flatten()
    grds[i, :] = gS3_x.flatten()
    acs_neigh[i, :] = np.real(S3_x_neigh).flatten()
    grds_neigh[i, :] = gS3_x_neigh.flatten()
