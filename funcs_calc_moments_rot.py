# -*- coding: utf-8 -*-
"""
Created on Sat Jul 18 12:23:14 2020

@author: kreym
"""

import numpy as np
import itertools
import scipy.special as special
import scipy.sparse as sp


import multiprocessing as mp
from psf_functions_2d import evaluate_psf_full
import psf_functions_2d

def calck1k2k3(L):
    kmap = list(itertools.product(np.arange(2*L-1), np.arange(2*L-1)))
    k2map = list(itertools.product(kmap, kmap))
    k1k2k3_map = []
    for k2 in k2map:
        shift1 = k2[0]
        shift2 = k2[1]
        idx3x = (-shift1[0] - shift2[0]) % (2*L - 1)
        idx3y = (-shift1[1] - shift2[1]) % (2*L - 1)
        shift3 = (idx3x, idx3y)
        k1k2k3_map.append((shift1, shift2, shift3))
    
    return k1k2k3_map

def calcmap3(L):
    k1k2k3_map = calck1k2k3(L)
    map3 = np.zeros((np.shape(k1k2k3_map)[0], 6))

    for i in range(np.shape(map3)[0]):
        map3[i, :] = [k1k2k3_map[i][0][0], k1k2k3_map[i][0][1], k1k2k3_map[i][1][0], k1k2k3_map[i][1][1], k1k2k3_map[i][2][0], k1k2k3_map[i][2][1]]
    return map3.astype(int)

def calck1k2k3_binned(L, R):
    equi_dict = {}
    for i1 in range(2*L-1):
        for j1 in range(2*L-1):
            for i2 in range(2*L-1):
                for j2 in range(2*L-1):
                    tmp = u_R((i1, j1), (i2, j2), R, L)
                    if tmp in equi_dict:
                        equi_dict[tmp].append(((i1, j1), (i2, j2)))
                    else:
                        equi_dict[tmp] = [((i1, j1), (i2, j2))]
    ks_list = list(equi_dict.values())
    k1k2_map_binned = [ks[0] for ks in ks_list]
    k1k2k3_map_binned = []
    for k2 in k1k2_map_binned:
        shift1 = k2[0]
        shift2 = k2[1]
        idx3x = (-shift1[0] - shift2[0]) % (2*L - 1)
        idx3y = (-shift1[1] - shift2[1]) % (2*L - 1)
        shift3 = (idx3x, idx3y)
        k1k2k3_map_binned.append((shift1, shift2, shift3))
    k1k2k3_map_equi = [ks[1:] for ks in ks_list]
    k1k2k3_map_equi_idxs = [[[k[i][j] for k in ks] for i in range(2) for j in range(2)] for ks in k1k2k3_map_equi]
    return k1k2k3_map_binned, k1k2k3_map_equi_idxs
         
def u_R(k1, k2, R, L):
    k1_abs = np.linalg.norm(k1)
    k2_abs = np.linalg.norm(k2)
    if k1_abs == 0 or k2_abs == 0:
        theta_k1k2 = 0
    else:
        theta_k1k2 = np.arccos(np.clip(np.dot(k1/k1_abs, k2/k2_abs), -1.0, 1.0))
    return np.floor(k1_abs*R / (2*L)) + R*np.floor(k2_abs*R / (2*L)) + (R**2) * np.floor(theta_k1k2*R / np.pi)

def calck1(L):
    k1_map = list(itertools.product(np.arange(2*L-1), np.arange(2*L-1)))

    return k1_map

def calcN_mat(L):
    N_mat = np.zeros((L, L, L, L))
    for ii in range(L):
        for jj in range(L):
            N_mat[ii, jj, ii, jj] += 1
    N_mat[ 0, 0, :, :] += 1
    N_mat[ :, :, 0, 0] += 1
    
    return sp.csr_matrix(np.reshape(N_mat, (L**4, 1)))

def calcS3_x(L, Nmax, Bk, z, kvals, k1k2k3_map):
    S3_k = np.zeros((2*L-1, 2*L-1, 2*L-1, 2*L-1), dtype=np.complex_)

    for k in range(Nmax):
        phi_nu = 2*np.pi * k / Nmax
        # omega_phi_nu = B * np.exp(1j*kvals*phi_nu)
        # f = np.pad(np.reshape(omega_phi_nu@z, (L, L)), L//2)
        # Fk = np.fft.fftn(np.fft.ifftshift(f))
        omega_phi_nu = Bk*np.exp(1j*kvals*phi_nu)
        Fk = omega_phi_nu @ z
        for k1k2k3 in k1k2k3_map:
            k1 = k1k2k3[0]
            k2 = k1k2k3[1]
            k3 = k1k2k3[2]
            S3_k[k1[0], k1[1], k2[0], k2[1]] += Fk[k1[0], k1[1]] * Fk[k2[0], k2[1]] * Fk[k3[0], k3[1]]
            
    S3_x = np.real(np.fft.ifftn(S3_k / Nmax)) / (L**2)
    return S3_x

def calcS3_x_neigh(L, Nmax, Bk, z, kvals, k1k2k3_map):
    S3_k_neigh = np.zeros((2*L-1, 2*L-1, 2*L-1, 2*L-1), dtype=np.complex_)
    
    # omega0 = B[:, kvals==0]
    # f0 = np.pad(np.reshape(omega0@z0, (L, L)), L//2)
    # F0 = np.fft.fftn(np.fft.ifftshift(f0))
    # for k in range(Nmax):
    #     phi_nu = 2*np.pi * k / Nmax
    #     omega_phi_nu = B * np.exp(1j*kvals*phi_nu)
    #     f = np.pad(np.reshape(omega_phi_nu@z, (L, L)), L//2)
    #     Fk = np.fft.fftn(np.fft.ifftshift(f))
        
    #     for k1k2k3 in k1k2k3_map:
    #         k1 = k1k2k3[0]
    #         k2 = k1k2k3[1]
    #         k3 = k1k2k3[2]
    #         S3_k_neigh[k1[0], k1[1], k2[0], k2[1]] += Fk[k1[0], k1[1]] * F0[k2[0], k2[1]] * Fk[k3[0], k3[1]]
            
    # S3_x_neigh = np.real(np.fft.ifftn(S3_k_neigh / Nmax)) / (L**2)
    # return S3_x_neigh
    
    omega0 = Bk*(kvals == 0)
    F0 = omega0 @ z
    for k in range(Nmax):
        phi_nu = 2*np.pi * k / Nmax
        omega_phi_nu = Bk*np.exp(1j*kvals*phi_nu)
        Fk = omega_phi_nu @ z
        
        for k1k2k3 in k1k2k3_map:
            k1 = k1k2k3[0]
            k2 = k1k2k3[1]
            k3 = k1k2k3[2]
            S3_k_neigh[k1[0], k1[1], k2[0], k2[1]] += Fk[k1[0], k1[1]] * F0[k2[0], k2[1]] * Fk[k3[0], k3[1]]
            
    S3_x_neigh = np.real(np.fft.ifftn(S3_k_neigh)) / (Nmax * L**2)
    return S3_x_neigh

def calcS3_x_grad(L, Nmax, Bk, z, kvals, k1k2k3_map):
    S3_k = np.zeros((2*L-1, 2*L-1, 2*L-1, 2*L-1), dtype=np.complex_)
    gS3_k = np.zeros((2*L-1, 2*L-1, 2*L-1, 2*L-1, len(z)), dtype=np.complex_)
    
    for k in range(Nmax):
        phi_nu = 2*np.pi * k / Nmax
        omega_phi_nu = Bk*np.exp(1j*kvals*phi_nu)
        Fk = omega_phi_nu @ z
        for k1k2k3 in k1k2k3_map:
            k1 = k1k2k3[0]
            k2 = k1k2k3[1]
            k3 = k1k2k3[2]
            S3_k[k1[0], k1[1], k2[0], k2[1]] += Fk[k1[0], k1[1]] * Fk[k2[0], k2[1]] * Fk[k3[0], k3[1]]
            gS3_k[k1[0], k1[1], k2[0], k2[1], :] += Fk[k1[0], k1[1]] * Fk[k2[0], k2[1]] * omega_phi_nu[k3[0], k3[1], :] +\
                Fk[k1[0], k1[1]] * omega_phi_nu[k2[0], k2[1], :] * Fk[k3[0], k3[1]] +\
                    omega_phi_nu[k1[0], k1[1], :] * Fk[k2[0], k2[1]] * Fk[k3[0], k3[1]]
    S3_xold = np.fft.ifftn(S3_k) / (Nmax * L**2)
    gS3_xold = np.fft.ifftn(gS3_k, axes=(0,1,2,3)) / (Nmax * L**2)
    return S3_xold, gS3_xold

def calcS3_x_gradnew(L, Nmax, Bk, z, kvals, k1k2k3_map):
    kvals = np.atleast_2d(kvals)
    S3_k = np.zeros((2*L-1, 2*L-1, 2*L-1, 2*L-1), dtype=np.complex_)
    gS3_k = np.zeros((2*L-1, 2*L-1, 2*L-1, 2*L-1, len(z)), dtype=np.complex_)
    
    k = np.atleast_2d(2*np.pi * np.arange(Nmax) / Nmax)
    omega_phi_nu = np.moveaxis(np.repeat(Bk[ :, :, :, np.newaxis], Nmax, axis=3)*np.exp(1j*kvals.T*k), [2, 3], [3, 2])
    Fk = omega_phi_nu @ z
    for k1k2k3 in k1k2k3_map:
        k1 = k1k2k3[0]
        k2 = k1k2k3[1]
        k3 = k1k2k3[2]
        S3_k[k1[0], k1[1], k2[0], k2[1]] = np.sum(Fk[k1[0], k1[1]] * Fk[k2[0], k2[1]] * Fk[k3[0], k3[1]])
        gS3_k[k1[0], k1[1], k2[0], k2[1], :] = np.sum(np.atleast_2d(Fk[k1[0], k1[1]] * Fk[k2[0], k2[1]]).T * omega_phi_nu[k3[0], k3[1], :, :] +\
            np.atleast_2d(Fk[k1[0], k1[1]] * Fk[k3[0], k3[1]]).T * omega_phi_nu[k2[0], k2[1], :, :] +\
                 np.atleast_2d(Fk[k2[0], k2[1]] * Fk[k3[0], k3[1]]).T * omega_phi_nu[k1[0], k1[1], :, :], axis=0)
    S3_xnew = np.fft.ifftn(S3_k) / (Nmax * L**2)
    gS3_xnew = np.fft.ifftn(gS3_k, axes=(0,1,2,3)) / (Nmax * L**2)
    return S3_xnew, gS3_xnew

def calcS3_x_gradnew2(L, Nmax, Bk, z, kvals, map3):
    kvals = np.atleast_2d(kvals)

    k = np.atleast_2d(2*np.pi * np.arange(Nmax) / Nmax)
    omega_phi_nu = np.moveaxis(np.repeat(Bk[ :, :, :, np.newaxis], Nmax, axis=3)*np.exp(1j*kvals.T*k), [2, 3], [3, 2])
    Fk = np.dot(omega_phi_nu, z)
    Fk0 = Fk[map3[ :, 0], map3[ :, 1], :]
    Fk1 = Fk[map3[ :, 2], map3[ :, 3], :]
    Fk2 = Fk[map3[ :, 4], map3[ :, 5], :]
    S3_k = np.reshape(np.sum(Fk0 * Fk1 * Fk2, axis=1), (2*L-1, 2*L-1, 2*L-1, 2*L-1))
    
    omega_phi_nu0 = omega_phi_nu[map3[ :, 0], map3[ :, 1], :, :]
    omega_phi_nu1 = omega_phi_nu[map3[ :, 2], map3[ :, 3], :, :]
    omega_phi_nu2 = omega_phi_nu[map3[ :, 4], map3[ :, 5], :, :]
    
    Fk0 = np.repeat(Fk0[ :, :, np.newaxis], len(z), axis=2)
    Fk1 = np.repeat(Fk1[ :, :, np.newaxis], len(z), axis=2)
    Fk2 = np.repeat(Fk2[ :, :, np.newaxis], len(z), axis=2)
    gS3_k = np.reshape(np.sum(Fk0 * Fk1 * omega_phi_nu2 + Fk0 * Fk2 * omega_phi_nu1 + Fk1 * Fk2 * omega_phi_nu0, axis=1), (2*L-1, 2*L-1, 2*L-1, 2*L-1, len(z)))

    S3_xnew2 = np.fft.ifftn(S3_k) / (Nmax * L**2)
    gS3_xnew2 = np.fft.ifftn(gS3_k, axes=(0,1,2,3)) / (Nmax * L**2)
    return S3_xnew2, gS3_xnew2

def calcS3_x_grad_k(L, Nmax, Bk, z, kvals, k1k2k3_map):
    S3_k = np.zeros((2*L-1, 2*L-1, 2*L-1, 2*L-1), dtype=np.complex_)
    gS3_k = np.zeros((2*L-1, 2*L-1, 2*L-1, 2*L-1, len(z)), dtype=np.complex_)
    
    for k in range(Nmax):
        phi_nu = 2*np.pi * k / Nmax
        omega_phi_nu = Bk*np.exp(1j*kvals*phi_nu)
        Fk = omega_phi_nu @ z
        for k1k2k3 in k1k2k3_map:
            k1 = k1k2k3[0]
            k2 = k1k2k3[1]
            k3 = k1k2k3[2]
            S3_k[k1[0], k1[1], k2[0], k2[1]] += Fk[k1[0], k1[1]] * Fk[k2[0], k2[1]] * Fk[k3[0], k3[1]]
            gS3_k[k1[0], k1[1], k2[0], k2[1], :] += Fk[k1[0], k1[1]] * Fk[k2[0], k2[1]] * omega_phi_nu[k3[0], k3[1], :] +\
                Fk[k1[0], k1[1]] * omega_phi_nu[k2[0], k2[1], :] * Fk[k3[0], k3[1]] +\
                    omega_phi_nu[k1[0], k1[1], :] * Fk[k2[0], k2[1]] * Fk[k3[0], k3[1]]
    S3_x = S3_k / (Nmax * L**2)
    gS3_x = gS3_k / (Nmax * L**2)
    return S3_x, gS3_x

def calcS3_x_grad_binned(L, Nmax, Bk, z, kvals, k1k2k3_map_binned, k1k2k3_map_equi_idxs):
    S3_k = np.zeros((2*L-1, 2*L-1, 2*L-1, 2*L-1), dtype=np.complex_)
    gS3_k = np.zeros((2*L-1, 2*L-1, 2*L-1, 2*L-1, len(z)), dtype=np.complex_)
    
    for k in range(Nmax):
        phi_nu = 2*np.pi * k / Nmax
        omega_phi_nu = Bk*np.exp(1j*kvals*phi_nu)
        Fk = omega_phi_nu @ z
        for k1k2k3 in k1k2k3_map_binned:
            k1 = k1k2k3[0]
            k2 = k1k2k3[1]
            k3 = k1k2k3[2]
            S3_k[k1[0], k1[1], k2[0], k2[1]] += Fk[k1[0], k1[1]] * Fk[k2[0], k2[1]] * Fk[k3[0], k3[1]]
            gS3_k[k1[0], k1[1], k2[0], k2[1], :] += Fk[k1[0], k1[1]] * Fk[k2[0], k2[1]] * omega_phi_nu[k3[0], k3[1], :] + Fk[k1[0], k1[1]] * omega_phi_nu[k2[0], k2[1], :] * Fk[k3[0], k3[1]] + omega_phi_nu[k1[0], k1[1], :] * Fk[k2[0], k2[1]] * Fk[k3[0], k3[1]]
            
    for (ii, k1k2k3) in enumerate(k1k2k3_map_binned):
        k1 = k1k2k3[0]
        k2 = k1k2k3[1]
        idxs = k1k2k3_map_equi_idxs[ii]
        S3_k[idxs[0], idxs[1], idxs[2], idxs[3]] = S3_k[k1[0], k1[1], k2[0], k2[1]]
        gS3_k[idxs[0], idxs[1], idxs[2], idxs[3], :] = gS3_k[k1[0], k1[1], k2[0], k2[1], :]
    S3_x = np.fft.ifftn(S3_k) / (Nmax * L**2)
    gS3_x = np.fft.ifftn(gS3_k, axes=(0,1,2,3)) / (Nmax * L**2)
    return S3_x, gS3_x

def calcS3_x_neigh_grad(L, Nmax, Bk, z, kvals, k1k2k3_map):
    S3_k_neigh = np.zeros((2*L-1, 2*L-1, 2*L-1, 2*L-1), dtype=np.complex_)
    gS3_k_neigh = np.zeros((2*L-1, 2*L-1, 2*L-1, 2*L-1, len(z)), dtype=np.complex_)
    
    omega0 = Bk * (kvals == 0)
    F0 = omega0 @ z
    for k in range(Nmax):
        phi_nu = 2*np.pi * k / Nmax
        omega_phi_nu = Bk*np.exp(1j*kvals*phi_nu)
        Fk = omega_phi_nu @ z
        
        for k1k2k3 in k1k2k3_map:
            k1 = k1k2k3[0]
            k2 = k1k2k3[1]
            k3 = k1k2k3[2]
            S3_k_neigh[k1[0], k1[1], k2[0], k2[1]] += Fk[k1[0], k1[1]] * F0[k2[0], k2[1]] * Fk[k3[0], k3[1]]
            gS3_k_neigh[k1[0], k1[1], k2[0], k2[1], :] += Fk[k1[0], k1[1]] * F0[k2[0], k2[1]] * omega_phi_nu[k3[0], k3[1], :] + Fk[k1[0], k1[1]] * omega0[k2[0], k2[1], :] * Fk[k3[0], k3[1]] + omega_phi_nu[k1[0], k1[1], :] * F0[k2[0], k2[1]] * Fk[k3[0], k3[1]]
      
    S3_x_neigh = np.fft.ifftn(S3_k_neigh) / (Nmax * L**2)
    gS3_x_neigh = np.fft.ifftn(gS3_k_neigh, axes=(0,1,2,3)) / (Nmax * L**2)
    return S3_x_neigh, gS3_x_neigh

def calcS3_x_neigh_gradnew2(L, Nmax, Bk, z, kvals, map3):
    kvals = np.atleast_2d(kvals)
    
    # %% Rotationally-averaged third-order autocorrelation with a neighbor
    Nmax0 = int((2/3)*Nmax)
    omega0 = Bk * (kvals == 0)
    omega0_expanded = np.moveaxis(np.repeat(omega0[:, :, :, np.newaxis], Nmax0, axis=3), [2, 3], [3, 2])
    F0 = np.dot(omega0, z)
    F0_expanded = np.repeat(F0[:, :, np.newaxis], Nmax0, axis=2)
    k = np.atleast_2d(2*np.pi * np.arange(Nmax0) / Nmax0)
    omega_phi_nu = np.moveaxis(np.repeat(Bk[ :, :, :, np.newaxis], Nmax0, axis=3)*np.exp(1j*kvals.T*k), [2, 3], [3, 2])
    Fk = np.dot(omega_phi_nu, z)
    Fk0 = Fk[map3[ :, 0], map3[ :, 1], :]
    Fk1 = F0_expanded[map3[ :, 2], map3[ :, 3], :]
    Fk2 = Fk[map3[ :, 4], map3[ :, 5], :]
    S3_k_neigh = np.reshape(np.sum(Fk0 * Fk1 * Fk2, axis=1), (2*L-1, 2*L-1, 2*L-1, 2*L-1))
    
    omega_phi_nu0 = omega_phi_nu[map3[ :, 0], map3[ :, 1], :, :]
    omega_phi_nu1 = omega0_expanded[map3[ :, 2], map3[ :, 3], :, :]
    omega_phi_nu2 = omega_phi_nu[map3[ :, 4], map3[ :, 5], :, :]
    
    Fk0 = np.repeat(Fk0[ :, :, np.newaxis], len(z), axis=2)
    Fk1 = np.repeat(Fk1[ :, :, np.newaxis], len(z), axis=2)
    Fk2 = np.repeat(Fk2[ :, :, np.newaxis], len(z), axis=2)
    gS3_k_neigh = np.reshape(np.sum(Fk0 * Fk1 * omega_phi_nu2 + Fk0 * Fk2 * omega_phi_nu1 + Fk1 * Fk2 * omega_phi_nu0, axis=1), (2*L-1, 2*L-1, 2*L-1, 2*L-1, len(z)))
    
    S3_x_neigh = np.fft.ifftn(S3_k_neigh) / (Nmax0 * L**2)
    gS3_x_neigh = np.fft.ifftn(gS3_k_neigh, axes=(0,1,2,3)) / (Nmax0 * L**2)
    return S3_x_neigh, gS3_x_neigh

def calcS3_x_triplets_gradnew2(L, Nmax, Bk, z, kvals, map3):
    kvals = np.atleast_2d(kvals)
    
    omega0 = Bk * (kvals == 0)
    F0 = np.dot(omega0, z)
    # %% Rotationally-averaged third-order autocorrelation with two neighbors
    Fk0 = np.atleast_2d(F0[map3[ :, 0], map3[ :, 1]])
    Fk1 = np.atleast_2d(F0[map3[ :, 2], map3[ :, 3]])
    Fk2 = np.atleast_2d(F0[map3[ :, 4], map3[ :, 5]])
    S3_k_triplets = np.reshape(np.squeeze(Fk0 * Fk1 * Fk2), (2*L-1, 2*L-1, 2*L-1, 2*L-1))
    
    omega_phi_nu0 = omega0[map3[ :, 0], map3[ :, 1], :]
    omega_phi_nu1 = omega0[map3[ :, 2], map3[ :, 3], :]
    omega_phi_nu2 = omega0[map3[ :, 4], map3[ :, 5], :]
    
    Fk0 = np.repeat(Fk0[ :, :, np.newaxis], len(z), axis=2)
    Fk1 = np.repeat(Fk1[ :, :, np.newaxis], len(z), axis=2)
    Fk2 = np.repeat(Fk2[ :, :, np.newaxis], len(z), axis=2)
    gS3_k_triplets = np.reshape(np.squeeze(Fk0 * Fk1 * omega_phi_nu2 + Fk0 * Fk2 * omega_phi_nu1 + Fk1 * Fk2 * omega_phi_nu0), (2*L-1, 2*L-1, 2*L-1, 2*L-1, len(z)))
    S3_x_triplets = np.fft.ifftn(S3_k_triplets) / (L**2)
    gS3_x_triplets = np.fft.ifftn(gS3_k_triplets, axes=(0,1,2,3)) / (L**2)
    return S3_x_triplets, gS3_x_triplets

def calcS3_x_triplets_grad(L, Nmax, Bk, z, kvals, k1k2k3_map):
    S3_k_triplets = np.zeros((2*L-1, 2*L-1, 2*L-1, 2*L-1), dtype=np.complex_)
    gS3_k_triplets = np.zeros((2*L-1, 2*L-1, 2*L-1, 2*L-1, len(z)), dtype=np.complex_)
    
    omega0 = Bk * (kvals == 0)
    F0 = omega0 @ z

    for k1k2k3 in k1k2k3_map:
        k1 = k1k2k3[0]
        k2 = k1k2k3[1]
        k3 = k1k2k3[2]
        S3_k_triplets[k1[0], k1[1], k2[0], k2[1]] += F0[k1[0], k1[1]] * F0[k2[0], k2[1]] * F0[k3[0], k3[1]]
        gS3_k_triplets[k1[0], k1[1], k2[0], k2[1], :] += F0[k1[0], k1[1]] * F0[k2[0], k2[1]] * omega0[k3[0], k3[1], :] + F0[k1[0], k1[1]] * omega0[k2[0], k2[1], :] * F0[k3[0], k3[1]] + omega0[k1[0], k1[1], :] * F0[k2[0], k2[1]] * F0[k3[0], k3[1]]
      
    S3_x_triplets = np.fft.ifftn(S3_k_triplets) / (L**2)
    gS3_x_triplets = np.fft.ifftn(gS3_k_triplets, axes=(0,1,2,3)) / (L**2)
    return S3_x_triplets, gS3_x_triplets

def calcS3_x_neigh_grad_binned(L, Nmax, Bk, z, kvals, k1k2k3_map_binned, k1k2k3_map_equi_idxs):
    S3_k_neigh = np.zeros((2*L-1, 2*L-1, 2*L-1, 2*L-1), dtype=np.complex_)
    gS3_k_neigh = np.zeros((2*L-1, 2*L-1, 2*L-1, 2*L-1, len(z)), dtype=np.complex_)
    
    omega0 = Bk * (kvals == 0)
    F0 = omega0 @ z
    for k in range(Nmax):
        phi_nu = 2*np.pi * k / Nmax
        omega_phi_nu = Bk*np.exp(1j*kvals*phi_nu)
        Fk = omega_phi_nu @ z
        
        for k1k2k3 in k1k2k3_map_binned:
            k1 = k1k2k3[0]
            k2 = k1k2k3[1]
            k3 = k1k2k3[2]
            S3_k_neigh[k1[0], k1[1], k2[0], k2[1]] += Fk[k1[0], k1[1]] * F0[k2[0], k2[1]] * Fk[k3[0], k3[1]]
            gS3_k_neigh[k1[0], k1[1], k2[0], k2[1], :] += Fk[k1[0], k1[1]] * F0[k2[0], k2[1]] * omega_phi_nu[k3[0], k3[1], :] + Fk[k1[0], k1[1]] * omega0[k2[0], k2[1], :] * Fk[k3[0], k3[1]] + omega_phi_nu[k1[0], k1[1], :] * F0[k2[0], k2[1]] * Fk[k3[0], k3[1]]
    
    for (ii, k1k2k3) in enumerate(k1k2k3_map_binned):
        k1 = k1k2k3[0]
        k2 = k1k2k3[1]
        idxs = k1k2k3_map_equi_idxs[ii]
        S3_k_neigh[idxs[0], idxs[1], idxs[2], idxs[3]] = S3_k_neigh[k1[0], k1[1], k2[0], k2[1]]
        gS3_k_neigh[idxs[0], idxs[1], idxs[2], idxs[3], :] = gS3_k_neigh[k1[0], k1[1], k2[0], k2[1], :]
    
    S3_x_neigh = np.fft.ifftn(S3_k_neigh) / (Nmax * L**2)
    gS3_x_neigh = np.fft.ifftn(gS3_k_neigh, axes=(0,1,2,3)) / (Nmax * L**2)
    return S3_x_neigh, gS3_x_neigh

def calcS3_full_shift(shift1, shift2, S3_x, S3_x_neigh, L, psf):
    shift1y = shift1[0]
    shift1x = shift1[1]
    
    shift2y = shift2[0]
    shift2x = shift2[1]
            
    # %% 1.
    M3_clean = S3_x[shift1y, shift1x, shift2y, shift2x]
    
    # %% 2. ->
    M3k_extra = 0
    
    # %% 2. 
    for j in range(max(shift1y, shift2y) - (L-1), L + min(shift1y, shift2y)):
        for i in range(max(shift1x, shift2x) - (L-1), L + min(shift1x, shift2x)):
            if not (np.abs(i) < L and np.abs(j) < L):
                M3k_extra = M3k_extra + evaluate_psf_full(psf, (j,i), L) * S3_x_neigh[shift2y-shift1y, shift2x-shift1x, j-shift1y, i-shift1x]
           
    # %% 3. 
    for j in range(shift1y - (L-1), L + shift1y - shift2y):
        for i in range(shift1x - (L-1), L + shift1x - shift2x):
            if not (np.abs(i) < L and np.abs(j) < L):
                M3k_extra = M3k_extra + evaluate_psf_full(psf, (j,i), L) * S3_x_neigh[j-shift1y, i-shift1x, j+shift2y-shift1y, i+shift2x-shift1x]
    
    # %% 4. 
    for j in range(shift2y - (L-1), L + shift2y - shift1y):
        for i in range(shift2x - (L-1), L + shift2x - shift1x):
            if not (np.abs(i) < L and np.abs(j) < L):
                M3k_extra = M3k_extra + evaluate_psf_full(psf, (j,i), L) * S3_x_neigh[j-shift2y, i-shift2x, j+shift1y-shift2y, i+shift1x-shift2x]
            
    M3 = M3_clean + M3k_extra
    return M3, M3_clean, M3k_extra

def calcS3_grad_full_shift(shift1, shift2, S3_x, gS3_x, S3_x_neigh, gS3_x_neigh, L, psf):
    shift1y = shift1[0]
    shift1x = shift1[1]
    
    shift2y = shift2[0]
    shift2x = shift2[1]
            
    # %% 1.
    M3_clean = S3_x[shift1y, shift1x, shift2y, shift2x]
    T3_clean = gS3_x[shift1y, shift1x, shift2y, shift2x]
    
    # %% 2. ->
    M3k_extra = 0
    T3k_extra = np.zeros_like(T3_clean)
    
    # %% 2. 
    for j in range(max(shift1y, shift2y) - (L-1), L + min(shift1y, shift2y)):
        for i in range(max(shift1x, shift2x) - (L-1), L + min(shift1x, shift2x)):
            if not (np.abs(i) < L and np.abs(j) < L):
                M3k_extra = M3k_extra + evaluate_psf_full(psf, (j,i), L) * S3_x_neigh[shift2y-shift1y, shift2x-shift1x, j-shift1y, i-shift1x]
                T3k_extra = T3k_extra + evaluate_psf_full(psf, (j,i), L) * gS3_x_neigh[shift2y-shift1y, shift2x-shift1x, j-shift1y, i-shift1x]
           
    # # %% 3. 
    # for j in range(shift1y - (L-1), L + shift1y - shift2y):
    #     for i in range(shift1x - (L-1), L + shift1x - shift2x):
    #         if not (np.abs(i) < L and np.abs(j) < L):
    #             M3k_extra = M3k_extra + evaluate_psf_full(psf, (j,i), L) * S3_x_neigh[j-shift1y, i-shift1x, j+shift2y-shift1y, i+shift2x-shift1x]
    #             T3k_extra = T3k_extra + evaluate_psf_full(psf, (j,i), L) * gS3_x_neigh[j-shift1y, i-shift1x, j+shift2y-shift1y, i+shift2x-shift1x]
    
    # # %% 4. 
    # for j in range(shift2y - (L-1), L + shift2y - shift1y):
    #     for i in range(shift2x - (L-1), L + shift2x - shift1x):
    #         if not (np.abs(i) < L and np.abs(j) < L):
    #             M3k_extra = M3k_extra + evaluate_psf_full(psf, (j,i), L) * S3_x_neigh[j-shift2y, i-shift2x, j+shift1y-shift2y, i+shift1x-shift2x]
    #             T3k_extra = T3k_extra + evaluate_psf_full(psf, (j,i), L) * gS3_x_neigh[j-shift2y, i-shift2x, j+shift1y-shift2y, i+shift1x-shift2x]
            
    # %% 3. 
    for j in range(shift1y - (L-1), L + shift1y - shift2y):
        for i in range(shift1x - (L-1), L + shift1x - shift2x):
            if not (np.abs(i) < L and np.abs(j) < L):
                M3k_extra = M3k_extra + evaluate_psf_full(psf, (j,i), L) * S3_x_neigh[shift2y, shift2x, shift1y-j, shift1x-i]
                T3k_extra = T3k_extra + evaluate_psf_full(psf, (j,i), L) * gS3_x_neigh[shift2y, shift2x, shift1y-j, shift1x-i]
    
    # %% 4. 
    for j in range(shift2y - (L-1), L + shift2y - shift1y):
        for i in range(shift2x - (L-1), L + shift2x - shift1x):
            if not (np.abs(i) < L and np.abs(j) < L):
                M3k_extra = M3k_extra + evaluate_psf_full(psf, (j,i), L) * S3_x_neigh[shift1y, shift1x, shift2y-j, shift2x-i]
                T3k_extra = T3k_extra + evaluate_psf_full(psf, (j,i), L) * gS3_x_neigh[shift1y, shift1x, shift2y-j, shift2x-i]
            
    M3 = M3_clean + M3k_extra
    T3 = T3_clean + T3k_extra
    return M3, T3



def calcS2_x_grad(L, Nmax, Bk, z, kvals, k1_map):
    S2_k = np.zeros((2*L-1, 2*L-1), dtype=np.complex_)
    gS2_k = np.zeros((2*L-1, 2*L-1, len(z)), dtype=np.complex_)
    
    for k in range(Nmax):
        phi_nu = 2*np.pi * k / Nmax
        omega_phi_nu = Bk*np.exp(1j*kvals*phi_nu)
        Fk = omega_phi_nu @ z
        for k1 in k1_map:
            S2_k[k1[0], k1[1]] += np.abs(Fk[k1[0], k1[1]])**2
            gS2_k[k1[0], k1[1], :] += 2 * Fk[k1[0], k1[1]] * omega_phi_nu[(-k1[0]) % (2*L-1), (-k1[1]) % (2*L-1), :]          
    S2_x = np.fft.ifftn(S2_k) / (Nmax * L**2)
    gS2_x = np.fft.ifftn(gS2_k, axes=(0,1)) / (Nmax * L**2)
    return S2_x, gS2_x

def calcS2_x_grad_k(L, Nmax, Bk, z, kvals, k1_map):
    S2_k = np.zeros((2*L-1, 2*L-1), dtype=np.complex_)
    gS2_k = np.zeros((2*L-1, 2*L-1, len(z)), dtype=np.complex_)
    
    for k in range(Nmax):
        phi_nu = 2*np.pi * k / Nmax
        omega_phi_nu = Bk*np.exp(1j*kvals*phi_nu)
        Fk = omega_phi_nu @ z
        for k1 in k1_map:
            S2_k[k1[0], k1[1]] += np.abs(Fk[k1[0], k1[1]])**2
            gS2_k[k1[0], k1[1], :] += 2 * Fk[k1[0], k1[1]] * omega_phi_nu[(-k1[0]) % (2*L-1), (-k1[1]) % (2*L-1), :]          
    S2_x = S2_k / (Nmax * L**2)
    gS2_x = gS2_k / (Nmax * L**2)
    return S2_x, gS2_x

def calcS2_x_neigh_grad(L, Bk, z, kvals, k1_map):
    S2_k_neigh = np.zeros((2*L-1, 2*L-1), dtype=np.complex_)
    gS2_k_neigh = np.zeros((2*L-1, 2*L-1, len(z)), dtype=np.complex_)
    
    omega0 = Bk * (kvals == 0)
    F0 = omega0 @ z
    for k1 in k1_map:
        S2_k_neigh[k1[0], k1[1]] += F0[k1[0], k1[1]] * F0[(-k1[0]) % (2*L-1), (-k1[1]) % (2*L-1)]
        gS2_k_neigh[k1[0], k1[1], :] += F0[k1[0], k1[1]] * omega0[(-k1[0]) % (2*L-1), (-k1[1]) % (2*L-1), :] + F0[(-k1[0]) % (2*L-1), (-k1[1]) % (2*L-1)] * omega0[k1[0], k1[1], :]
    S2_x_neigh = np.fft.ifftn(S2_k_neigh) / (L**2)
    gS2_x_neigh = np.fft.ifftn(gS2_k_neigh, axes=(0,1)) / (L**2)
    return S2_x_neigh, gS2_x_neigh

def calcS2_grad_full_shift(shift1, S2_x, gS2_x, S2_x_neigh, gS2_x_neigh, L, psf):
    shift1x = shift1[0]
    shift1y = shift1[1]
    
    # %% 1.
    M2_clean = S2_x[shift1x, shift1y]
    T2_clean = gS2_x[shift1x, shift1y]
    
    M2k_extra = 0
    T2k_extra = np.zeros_like(T2_clean)
    
    # %% 2.
    for j in range(shift1y-(L-1), L + shift1y):
        for i in range(shift1x-(L-1), L + shift1x):
            if not (np.abs(i) < L and np.abs(j) < L):
                M2k_extra = M2k_extra + evaluate_psf_full(psf, (i, j), L) \
                * S2_x_neigh[i-shift1x, j-shift1y]
                T2k_extra = T2k_extra + evaluate_psf_full(psf, (i, j), L) \
                * gS2_x_neigh[i-shift1x, j-shift1y]
                
    M2 = M2_clean + M2k_extra
    T2 = T2_clean + T2k_extra
    
    return M2, T2

def calcS2_grad_full_shift_coeffs(shift1, S2_x, gS2_x, S2_x_neigh, gS2_x_neigh, L, psf_coeffs, roots, N_coeffs, b):
    shift1y = shift1[0]
    shift1x = shift1[1]
    # %% 1.
    M2_clean = S2_x[shift1y, shift1x]
    T2_clean = gS2_x[shift1y, shift1x]
    
    M2k_extra = 0
    T2k_extra = np.zeros_like(T2_clean)
    g_coeffs = np.zeros(np.shape(psf_coeffs), dtype=np.complex_)

    # %% 2.
    for j in range(shift1y-(L-1), L + shift1y):
        for i in range(shift1x-(L-1), L + shift1x):
            if not (np.abs(i) < L and np.abs(j) < L):
                psf_i_j = psf_functions_2d.fourier_bessel_expansion(psf_coeffs, b, np.sqrt(i**2 + j**2))
                Js = np.array([special.j0(roots[n]*np.sqrt(i**2 + j**2)/b) for n in range(len(psf_coeffs))])
                g_coeffs = g_coeffs + S2_x_neigh[j-shift1y, i-shift1x] * Js
                M2k_extra = M2k_extra + psf_i_j * S2_x_neigh[j-shift1y, i-shift1x]
                T2k_extra = T2k_extra + psf_i_j * gS2_x_neigh[j-shift1y, i-shift1x]
                
    M2 = M2_clean + M2k_extra
    T2 = T2_clean + T2k_extra
    
    return M2, T2, g_coeffs

def calcS3_grad_full_shift_coeffs(shift1, shift2, S3_x, gS3_x, S3_x_neigh, gS3_x_neigh, L, psf_coeffs, roots, N_coeffs, b):
    shift1y = shift1[0]
    shift1x = shift1[1]
    
    shift2y = shift2[0]
    shift2x = shift2[1]
            
    # %% 1.
    M3_clean = S3_x[shift1y, shift1x, shift2y, shift2x]
    T3_clean = gS3_x[shift1y, shift1x, shift2y, shift2x]
    
    # %% 2. ->
    M3k_extra = 0
    T3k_extra = np.zeros_like(T3_clean)
    g_coeffs = np.zeros(np.shape(psf_coeffs), dtype=np.complex_)

    # %% 2. 
    for j in range(max(shift1y, shift2y) - (L-1), L + min(shift1y, shift2y)):
        for i in range(max(shift1x, shift2x) - (L-1), L + min(shift1x, shift2x)):
            if not (np.abs(i) < L and np.abs(j) < L):
                psf_i_j = psf_functions_2d.fourier_bessel_expansion(psf_coeffs, b, np.sqrt(i**2 + j**2))
                Js = np.array([special.j0(roots[n]*np.sqrt(i**2 + j**2)/b) for n in range(N_coeffs)])
                g_coeffs = g_coeffs + S3_x_neigh[shift2y-shift1y, shift2x-shift1x, j-shift1y, i-shift1x] * Js
                M3k_extra = M3k_extra + psf_i_j * S3_x_neigh[shift2y-shift1y, shift2x-shift1x, j-shift1y, i-shift1x]
                T3k_extra = T3k_extra + psf_i_j * gS3_x_neigh[shift2y-shift1y, shift2x-shift1x, j-shift1y, i-shift1x]
           
    # %% 3. 
    for j in range(shift1y - (L-1), L + shift1y - shift2y):
        for i in range(shift1x - (L-1), L + shift1x - shift2x):
            if not (np.abs(i) < L and np.abs(j) < L):
                psf_i_j = psf_functions_2d.fourier_bessel_expansion(psf_coeffs, b, np.sqrt(i**2 + j**2))
                Js = np.array([special.j0(roots[n]*np.sqrt(i**2 + j**2)/b) for n in range(N_coeffs)])
                g_coeffs = g_coeffs + S3_x_neigh[shift2y, shift2x, shift1y-j, shift1x-i] * Js
                M3k_extra = M3k_extra + psf_i_j * S3_x_neigh[shift2y, shift2x, shift1y-j, shift1x-i]
                T3k_extra = T3k_extra + psf_i_j * gS3_x_neigh[shift2y, shift2x, shift1y-j, shift1x-i]
    
    # %% 4. 
    for j in range(shift2y - (L-1), L + shift2y - shift1y):
        for i in range(shift2x - (L-1), L + shift2x - shift1x):
            if not (np.abs(i) < L and np.abs(j) < L):
                psf_i_j = psf_functions_2d.fourier_bessel_expansion(psf_coeffs, b, np.sqrt(i**2 + j**2))
                Js = np.array([special.j0(roots[n]*np.sqrt(i**2 + j**2)/b) for n in range(N_coeffs)])
                g_coeffs = g_coeffs + S3_x_neigh[shift1y, shift1x, shift2y-j, shift2x-i] * Js
                M3k_extra = M3k_extra + psf_i_j * S3_x_neigh[shift1y, shift1x, shift2y-j, shift2x-i]
                T3k_extra = T3k_extra + psf_i_j * gS3_x_neigh[shift1y, shift1x, shift2y-j, shift2x-i]
                
    M3 = M3_clean + M3k_extra
    T3 = T3_clean + T3k_extra
    return M3, T3, g_coeffs


def calcoeffs2_grad(shift1, S2_x, gS2_x, S2_x_neigh, gS2_x_neigh, L, psf_coeffs, roots, N_coeffs, b):
    shift1y = shift1[0]
    shift1x = shift1[1]
    # %% 1.
    M2_clean = S2_x[shift1y, shift1x]
    T2_clean = gS2_x[shift1y, shift1x]
    
    M2k_extra = 0
    T2k_extra = np.zeros_like(T2_clean)
    g_coeffs = np.zeros(np.shape(psf_coeffs), dtype=np.complex_)

    # %% 2.
    for j in range(shift1y-(L-1), L + shift1y):
        for i in range(shift1x-(L-1), L + shift1x):
            if not (np.abs(i) < L and np.abs(j) < L):
                psf_i_j = psf_functions_2d.fourier_bessel_expansion(psf_coeffs, b, np.sqrt(i**2 + j**2))
                Js = np.array([special.j0(roots[n]*np.sqrt(i**2 + j**2)/b) for n in range(len(psf_coeffs))])
                g_coeffs = g_coeffs + S2_x_neigh[j-shift1y, i-shift1x] * Js
                
    M2 = M2_clean + M2k_extra
    T2 = T2_clean + T2k_extra
    
    return M2, T2, g_coeffs

def calccoeffs3_grad(shift1, shift2, S3_x, gS3_x, S3_x_neigh, gS3_x_neigh, L, psf_coeffs, roots, N_coeffs, b):
    shift1y = shift1[0]
    shift1x = shift1[1]
    
    shift2y = shift2[0]
    shift2x = shift2[1]
            
    # %% 1.
    
    # %% 2. ->
    g_coeffs = np.zeros(np.shape(psf_coeffs), dtype=np.complex_)

    # %% 2. 
    for j in range(max(shift1y, shift2y) - (L-1), L + min(shift1y, shift2y)):
        for i in range(max(shift1x, shift2x) - (L-1), L + min(shift1x, shift2x)):
            if not (np.abs(i) < L and np.abs(j) < L):
                psf_i_j = psf_functions_2d.fourier_bessel_expansion(psf_coeffs, b, np.sqrt(i**2 + j**2))
                Js = np.array([special.j0(roots[n]*np.sqrt(i**2 + j**2)/b) for n in range(N_coeffs)])
                g_coeffs = g_coeffs + S3_x_neigh[shift2y-shift1y, shift2x-shift1x, j-shift1y, i-shift1x] * Js
           
    # %% 3. 
    for j in range(shift1y - (L-1), L + shift1y - shift2y):
        for i in range(shift1x - (L-1), L + shift1x - shift2x):
            if not (np.abs(i) < L and np.abs(j) < L):
                psf_i_j = psf_functions_2d.fourier_bessel_expansion(psf_coeffs, b, np.sqrt(i**2 + j**2))
                Js = np.array([special.j0(roots[n]*np.sqrt(i**2 + j**2)/b) for n in range(N_coeffs)])
                g_coeffs = g_coeffs + S3_x_neigh[shift2y, shift2x, shift1y-j, shift1x-i] * Js
    
    # %% 4. 
    for j in range(shift2y - (L-1), L + shift2y - shift1y):
        for i in range(shift2x - (L-1), L + shift2x - shift1x):
            if not (np.abs(i) < L and np.abs(j) < L):
                psf_i_j = psf_functions_2d.fourier_bessel_expansion(psf_coeffs, b, np.sqrt(i**2 + j**2))
                Js = np.array([special.j0(roots[n]*np.sqrt(i**2 + j**2)/b) for n in range(N_coeffs)])
                g_coeffs = g_coeffs + S3_x_neigh[shift1y, shift1x, shift2y-j, shift2x-i] * Js

    return g_coeffs


# def calcS3_x_grad_parallel(L, Nmax, Bk, z, kvals, k1k2k3_map):
#     num_cpus = mp.cpu_count()
#     divided_ks = np.array_split(np.arange(Nmax), num_cpus)
    
#     pool = mp.Pool(num_cpus)
#     S = pool.starmap(calcS3_x_grad_partial, [[L, ks, Nmax, Bk, z, kvals, k1k2k3_map] for ks in divided_ks])
#     pool.close()
#     pool.join()
    
#     S3_x = np.sum(S3 for S3, gS3, S3_neigh, gS3_neigh in S)
#     gS3_x = np.sum(gS3 for S3, gS3, S3_neigh, gS3_neigh in S)
#     S3_x_neigh = np.sum(S3_neigh for S3, gS3, S3_neigh, gS3_neigh in S)
#     gS3_x_neigh = np.sum(gS3_neigh for S3, gS3, S3_neigh, gS3_neigh in S)
#     return S3_x, gS3_x, S3_x_neigh, gS3_x_neigh

# def calcS3_x_grad_partial(L, ks, Nmax, Bk, z, kvals, k1k2k3_map):
#     S3_k = np.zeros((2*L-1, 2*L-1, 2*L-1, 2*L-1), dtype=np.complex_)
#     gS3_k = np.zeros((2*L-1, 2*L-1, 2*L-1, 2*L-1, len(z)), dtype=np.complex_)
#     S3_k_neigh = np.zeros((2*L-1, 2*L-1, 2*L-1, 2*L-1), dtype=np.complex_)
#     gS3_k_neigh = np.zeros((2*L-1, 2*L-1, 2*L-1, 2*L-1, len(z)), dtype=np.complex_)
    
#     if ks != []:
#         if np.min(ks) < (2/3)*Nmax:
#             omega0 = Bk * (kvals == 0)
#             F0 = omega0 @ z
#             for k in ks:
#                 phi_nu = 2*np.pi * k / Nmax
#                 omega_phi_nu = Bk*np.exp(1j*kvals*phi_nu)
#                 Fk = omega_phi_nu @ z
#                 for k1k2k3 in k1k2k3_map:
#                     k1 = k1k2k3[0]
#                     k2 = k1k2k3[1]
#                     k3 = k1k2k3[2]
#                     S3_k[k1[0], k1[1], k2[0], k2[1]] += Fk[k1[0], k1[1]] * Fk[k2[0], k2[1]] * Fk[k3[0], k3[1]]
#                     gS3_k[k1[0], k1[1], k2[0], k2[1], :] += Fk[k1[0], k1[1]] * Fk[k2[0], k2[1]] * omega_phi_nu[k3[0], k3[1], :] +\
#                         Fk[k1[0], k1[1]] * omega_phi_nu[k2[0], k2[1], :] * Fk[k3[0], k3[1]] +\
#                             omega_phi_nu[k1[0], k1[1], :] * Fk[k2[0], k2[1]] * Fk[k3[0], k3[1]]
                    
#                     S3_k_neigh[k1[0], k1[1], k2[0], k2[1]] += Fk[k1[0], k1[1]] * F0[k2[0], k2[1]] * Fk[k3[0], k3[1]]
#                     gS3_k_neigh[k1[0], k1[1], k2[0], k2[1], :] += Fk[k1[0], k1[1]] * F0[k2[0], k2[1]] * omega_phi_nu[k3[0], k3[1], :] + Fk[k1[0], k1[1]] * omega0[k2[0], k2[1], :] * Fk[k3[0], k3[1]] + omega_phi_nu[k1[0], k1[1], :] * F0[k2[0], k2[1]] * Fk[k3[0], k3[1]]
#         else:
#             for k in ks:
#                 phi_nu = 2*np.pi * k / Nmax
#                 omega_phi_nu = Bk*np.exp(1j*kvals*phi_nu)
#                 Fk = omega_phi_nu @ z
#                 for k1k2k3 in k1k2k3_map:
#                     k1 = k1k2k3[0]
#                     k2 = k1k2k3[1]
#                     k3 = k1k2k3[2]
#                     S3_k[k1[0], k1[1], k2[0], k2[1]] += Fk[k1[0], k1[1]] * Fk[k2[0], k2[1]] * Fk[k3[0], k3[1]]
#                     gS3_k[k1[0], k1[1], k2[0], k2[1], :] += Fk[k1[0], k1[1]] * Fk[k2[0], k2[1]] * omega_phi_nu[k3[0], k3[1], :] +\
#                         Fk[k1[0], k1[1]] * omega_phi_nu[k2[0], k2[1], :] * Fk[k3[0], k3[1]] +\
#                             omega_phi_nu[k1[0], k1[1], :] * Fk[k2[0], k2[1]] * Fk[k3[0], k3[1]]
    
#     S3_x = np.fft.ifftn(S3_k) / (Nmax * L**2)
#     gS3_x = np.fft.ifftn(gS3_k, axes=(0,1,2,3)) / (Nmax * L**2)
#     S3_x_neigh = np.fft.ifftn(S3_k_neigh) / (Nmax * L**2)
#     gS3_x_neigh = np.fft.ifftn(gS3_k_neigh, axes=(0,1,2,3)) / (Nmax * L**2)
#     return S3_x, gS3_x, S3_x_neigh, gS3_x_neigh


def calcS3_x_grad_parallel_ver2(L, Nmax, Bk, z, kvals, k1k2k3_map):
    num_cpus = mp.cpu_count()
    divided_k1k2k3map = np.array_split(k1k2k3_map, num_cpus)
    
    pool = mp.Pool(num_cpus)
    S = pool.starmap(calcS3_x_grad_partial_ver2, [[L, Nmax, Bk, z, kvals, div_k1k2k3_map] for div_k1k2k3_map in divided_k1k2k3map])
    pool.close()
    pool.join()
    
    S3_k = np.sum(S3 for S3, gS3, S3_neigh, gS3_neigh in S)
    gS3_k = np.sum(gS3 for S3, gS3, S3_neigh, gS3_neigh in S)
    S3_k_neigh = np.sum(S3_neigh for S3, gS3, S3_neigh, gS3_neigh in S)
    gS3_k_neigh = np.sum(gS3_neigh for S3, gS3, S3_neigh, gS3_neigh in S)
    
    S3_x = np.fft.ifftn(S3_k) / (Nmax * L**2)
    gS3_x = np.fft.ifftn(gS3_k, axes=(0,1,2,3)) / (Nmax * L**2)
    S3_x_neigh = np.fft.ifftn(S3_k_neigh) / ((2/3)*Nmax * L**2)
    gS3_x_neigh = np.fft.ifftn(gS3_k_neigh, axes=(0,1,2,3)) / ((2/3)*Nmax * L**2)
    return S3_x, gS3_x, S3_x_neigh, gS3_x_neigh

def calcS3_x_grad_partial_ver2(L, Nmax, Bk, z, kvals, k1k2k3_map):
    
    
    S3_k = np.zeros((2*L-1, 2*L-1, 2*L-1, 2*L-1), dtype=np.complex_)
    gS3_k = np.zeros((2*L-1, 2*L-1, 2*L-1, 2*L-1, len(z)), dtype=np.complex_)
    S3_k_neigh = np.zeros((2*L-1, 2*L-1, 2*L-1, 2*L-1), dtype=np.complex_)
    gS3_k_neigh = np.zeros((2*L-1, 2*L-1, 2*L-1, 2*L-1, len(z)), dtype=np.complex_)
    
    # %% Rotationally-averaged third-order autocorrelation
    for k in range(Nmax):
        phi_nu = 2*np.pi * k / Nmax
        omega_phi_nu = Bk*np.exp(1j*kvals*phi_nu)
        Fk = omega_phi_nu @ z

        for k1k2k3 in k1k2k3_map:
            k1 = k1k2k3[0]
            k2 = k1k2k3[1]
            k3 = k1k2k3[2]
            S3_k[k1[0], k1[1], k2[0], k2[1]] += Fk[k1[0], k1[1]] * Fk[k2[0], k2[1]] * Fk[k3[0], k3[1]]
            gS3_k[k1[0], k1[1], k2[0], k2[1], :] += Fk[k1[0], k1[1]] * Fk[k2[0], k2[1]] * omega_phi_nu[k3[0], k3[1], :] +\
                Fk[k1[0], k1[1]] * omega_phi_nu[k2[0], k2[1], :] * Fk[k3[0], k3[1]] +\
                    omega_phi_nu[k1[0], k1[1], :] * Fk[k2[0], k2[1]] * Fk[k3[0], k3[1]]
    
    # %% Rotationally-averaged third-order autocorrelation with a neighbor
    omega0 = Bk * (kvals == 0)
    F0 = omega0 @ z
    Nmax0 = int((2/3)*Nmax)
    for k in range(Nmax0):
        phi_nu = 2*np.pi * k / Nmax0
        omega_phi_nu = Bk*np.exp(1j*kvals*phi_nu)
        Fk = omega_phi_nu @ z

        for k1k2k3 in k1k2k3_map:
            k1 = k1k2k3[0]
            k2 = k1k2k3[1]
            k3 = k1k2k3[2]
            S3_k_neigh[k1[0], k1[1], k2[0], k2[1]] += Fk[k1[0], k1[1]] * F0[k2[0], k2[1]] * Fk[k3[0], k3[1]]
            gS3_k_neigh[k1[0], k1[1], k2[0], k2[1], :] += Fk[k1[0], k1[1]] * F0[k2[0], k2[1]] * omega_phi_nu[k3[0], k3[1], :] + Fk[k1[0], k1[1]] * omega0[k2[0], k2[1], :] * Fk[k3[0], k3[1]] + omega_phi_nu[k1[0], k1[1], :] * F0[k2[0], k2[1]] * Fk[k3[0], k3[1]]
    
    return S3_k, gS3_k, S3_k_neigh, gS3_k_neigh

# class NDSparseMatrix:
#   def __init__(self):
#     self.elements = {}

#   def addValue(self, indexes, value):
#     self.elements[indexes] = value

#   def readValue(self, indexes):
#     try:
#       value = self.elements[indexes]
#     except KeyError:
#       # could also be 0.0 if using floats...
#       value = 0 + 0j
#     return value

def calcS3_x_grad_parallel(L, Nmax, Bk, z, kvals, k1k2k3_map):
    num_cpus = mp.cpu_count()
    divided_k1k2k3map = np.array_split(k1k2k3_map, num_cpus)
    
    pool = mp.Pool(num_cpus)
    S = pool.starmap(calcS3_x_grad_partial, [[L, Nmax, Bk, z, kvals, div_k1k2k3_map] for div_k1k2k3_map in divided_k1k2k3map])
    pool.close()
    pool.join()
    
    S3_k_tmp = [S3 for S3, gS3, S3_neigh, gS3_neigh in S]
    S3_k_dict = {k:v for el in S3_k_tmp for k,v in el.items()}
    S3_k = np.reshape(list(S3_k_dict.values()), (2*L-1, 2*L-1, 2*L-1, 2*L-1))
    
    gS3_k_tmp = [gS3 for S3, gS3, S3_neigh, gS3_neigh in S]
    gS3_k_dict = {k:v for el in gS3_k_tmp for k,v in el.items()}
    gS3_k = np.reshape(list(gS3_k_dict.values()), (2*L-1, 2*L-1, 2*L-1, 2*L-1, len(z)))
    
    S3_k_neigh_tmp = [S3_neigh for S3, gS3, S3_neigh, gS3_neigh in S]
    S3_k_neigh_dict = {k:v for el in S3_k_neigh_tmp for k,v in el.items()}
    S3_k_neigh = np.reshape(list(S3_k_neigh_dict.values()), (2*L-1, 2*L-1, 2*L-1, 2*L-1))
    
    gS3_k_neigh_tmp = [gS3_neigh for S3, gS3, S3_neigh, gS3_neigh in S]
    gS3_k_neigh_dict = {k:v for el in gS3_k_neigh_tmp for k,v in el.items()}
    gS3_k_neigh = np.reshape(list(gS3_k_neigh_dict.values()), (2*L-1, 2*L-1, 2*L-1, 2*L-1, len(z)))
    
    S3_x = np.fft.ifftn(S3_k) / (Nmax * L**2)
    gS3_x = np.fft.ifftn(gS3_k, axes=(0,1,2,3)) / (Nmax * L**2)
    S3_x_neigh = np.fft.ifftn(S3_k_neigh) / ((2/3)*Nmax * L**2)
    gS3_x_neigh = np.fft.ifftn(gS3_k_neigh, axes=(0,1,2,3)) / ((2/3)*Nmax * L**2)
    return S3_x, gS3_x, S3_x_neigh, gS3_x_neigh

def calcS3_x_grad_partial(L, Nmax, Bk, z, kvals, k1k2k3_map):
    S3_k = {}
    gS3_k = {}
    S3_k_neigh = {}
    gS3_k_neigh = {}
    
    for k1k2k3 in k1k2k3_map:
        k1 = k1k2k3[0]
        k2 = k1k2k3[1]
        k3 = k1k2k3[2]
        S3_k[(k1[0], k1[1], k2[0], k2[1])] = 0 + 0j
        gS3_k[(k1[0], k1[1], k2[0], k2[1])] = np.zeros((len(z), ), dtype=np.complex_)
        S3_k_neigh[(k1[0], k1[1], k2[0], k2[1])] = 0 + 0j
        gS3_k_neigh[(k1[0], k1[1], k2[0], k2[1])] = np.zeros((len(z), ), dtype=np.complex_)
    
    # %% Rotationally-averaged third-order autocorrelation
    for k in range(Nmax):
        phi_nu = 2*np.pi * k / Nmax
        omega_phi_nu = Bk*np.exp(1j*kvals*phi_nu)
        Fk = omega_phi_nu @ z

        for k1k2k3 in k1k2k3_map:
            k1 = k1k2k3[0]
            k2 = k1k2k3[1]
            k3 = k1k2k3[2]
            S3_k[(k1[0], k1[1], k2[0], k2[1])] += Fk[k1[0], k1[1]] * Fk[k2[0], k2[1]] * Fk[k3[0], k3[1]]
            gS3_k[(k1[0], k1[1], k2[0], k2[1])] += Fk[k1[0], k1[1]] * Fk[k2[0], k2[1]] * omega_phi_nu[k3[0], k3[1], :] +\
                Fk[k1[0], k1[1]] * omega_phi_nu[k2[0], k2[1], :] * Fk[k3[0], k3[1]] +\
                    omega_phi_nu[k1[0], k1[1], :] * Fk[k2[0], k2[1]] * Fk[k3[0], k3[1]]
    
    # %% Rotationally-averaged third-order autocorrelation with a neighbor
    omega0 = Bk * (kvals == 0)
    F0 = omega0 @ z
    Nmax0 = int((2/3)*Nmax)
    for k in range(Nmax0):
        phi_nu = 2*np.pi * k / Nmax0
        omega_phi_nu = Bk*np.exp(1j*kvals*phi_nu)
        Fk = omega_phi_nu @ z

        for k1k2k3 in k1k2k3_map:
            k1 = k1k2k3[0]
            k2 = k1k2k3[1]
            k3 = k1k2k3[2]
            S3_k_neigh[(k1[0], k1[1], k2[0], k2[1])] += Fk[k1[0], k1[1]] * F0[k2[0], k2[1]] * Fk[k3[0], k3[1]]
            gS3_k_neigh[(k1[0], k1[1], k2[0], k2[1])] += Fk[k1[0], k1[1]] * F0[k2[0], k2[1]] * omega_phi_nu[k3[0], k3[1], :] + Fk[k1[0], k1[1]] * omega0[k2[0], k2[1], :] * Fk[k3[0], k3[1]] + omega_phi_nu[k1[0], k1[1], :] * F0[k2[0], k2[1]] * Fk[k3[0], k3[1]]
    
    return S3_k, gS3_k, S3_k_neigh, gS3_k_neigh

def calcS3_x_grad_neigh_triplets_parallel(L, Nmax, Bk, z, kvals, map3):
    num_cpus = mp.cpu_count()
    divided_k1k2k3map = np.array_split(map3, num_cpus)
    
    pool = mp.Pool(num_cpus)
    S = pool.starmap(calcS3_x_grad_neigh_triplets_partial, [[L, Nmax, Bk, z, kvals, div_k1k2k3_map] for div_k1k2k3_map in divided_k1k2k3map])
    pool.close()
    pool.join()
    
    S3_k = np.reshape(np.concatenate([S3 for S3, gS3, S3_neigh, gS3_neigh, S3_triplets, gS3_triplets in S]), (2*L-1, 2*L-1, 2*L-1, 2*L-1))
    
    gS3_k = np.reshape(np.concatenate([gS3 for S3, gS3, S3_neigh, gS3_neigh, S3_triplets, gS3_triplets in S]), (2*L-1, 2*L-1, 2*L-1, 2*L-1, len(z)))

    S3_k_neigh = np.reshape(np.concatenate([S3_neigh for S3, gS3, S3_neigh, gS3_neigh, S3_triplets, gS3_triplets in S]), (2*L-1, 2*L-1, 2*L-1, 2*L-1))
    
    gS3_k_neigh = np.reshape(np.concatenate([gS3_neigh for S3, gS3, S3_neigh, gS3_neigh, S3_triplets, gS3_triplets in S]), (2*L-1, 2*L-1, 2*L-1, 2*L-1, len(z)))
    
    S3_k_triplets = np.reshape(np.concatenate([S3_triplets for S3, gS3, S3_neigh, gS3_neigh, S3_triplets, gS3_triplets in S]), (2*L-1, 2*L-1, 2*L-1, 2*L-1))
    
    gS3_k_triplets = np.reshape(np.concatenate([gS3_triplets for S3, gS3, S3_neigh, gS3_neigh, S3_triplets, gS3_triplets in S]), (2*L-1, 2*L-1, 2*L-1, 2*L-1, len(z)))
    
    S3_x = np.fft.ifftn(S3_k) / (Nmax * L**2)
    gS3_x = np.fft.ifftn(gS3_k, axes=(0,1,2,3)) / (Nmax * L**2)
    S3_x_neigh = np.fft.ifftn(S3_k_neigh) / ((2/3)*Nmax * L**2)
    gS3_x_neigh = np.fft.ifftn(gS3_k_neigh, axes=(0,1,2,3)) / ((2/3)*Nmax * L**2)
    S3_x_triplets = np.fft.ifftn(S3_k_triplets) / (L**2)
    gS3_x_triplets = np.fft.ifftn(gS3_k_triplets, axes=(0,1,2,3)) / (L**2)
    return S3_x, gS3_x, S3_x_neigh, gS3_x_neigh, S3_x_triplets, gS3_x_triplets

def calcS3_x_grad_neigh_triplets_partial(L, Nmax, Bk, z, kvals, map3):
    kvals = np.atleast_2d(kvals)
    # %% Rotationally-averaged third-order autocorrelation
    k = np.atleast_2d(2*np.pi * np.arange(Nmax) / Nmax)
    omega_phi_nu = np.moveaxis(np.repeat(Bk[ :, :, :, np.newaxis], Nmax, axis=3)*np.exp(1j*kvals.T*k), [2, 3], [3, 2])
    Fk = np.dot(omega_phi_nu, z)
    Fk0 = Fk[map3[ :, 0], map3[ :, 1], :]
    Fk1 = Fk[map3[ :, 2], map3[ :, 3], :]
    Fk2 = Fk[map3[ :, 4], map3[ :, 5], :]
    S3_k = np.sum(Fk0 * Fk1 * Fk2, axis=1)
    
    omega_phi_nu0 = omega_phi_nu[map3[ :, 0], map3[ :, 1], :, :]
    omega_phi_nu1 = omega_phi_nu[map3[ :, 2], map3[ :, 3], :, :]
    omega_phi_nu2 = omega_phi_nu[map3[ :, 4], map3[ :, 5], :, :]
    
    Fk0 = np.repeat(Fk0[ :, :, np.newaxis], len(z), axis=2)
    Fk1 = np.repeat(Fk1[ :, :, np.newaxis], len(z), axis=2)
    Fk2 = np.repeat(Fk2[ :, :, np.newaxis], len(z), axis=2)
    gS3_k = np.sum(Fk0 * Fk1 * omega_phi_nu2 + Fk0 * Fk2 * omega_phi_nu1 + Fk1 * Fk2 * omega_phi_nu0, axis=1)

    # %% Rotationally-averaged third-order autocorrelation with a neighbor
    Nmax0 = int((2/3)*Nmax)
    omega0 = Bk * (kvals == 0)
    omega0_expanded = np.moveaxis(np.repeat(omega0[:, :, :, np.newaxis], Nmax0, axis=3), [2, 3], [3, 2])
    F0 = np.dot(omega0, z)
    F0_expanded = np.repeat(F0[:, :, np.newaxis], Nmax0, axis=2)
    k = np.atleast_2d(2*np.pi * np.arange(Nmax0) / Nmax0)
    omega_phi_nu = np.moveaxis(np.repeat(Bk[ :, :, :, np.newaxis], Nmax0, axis=3)*np.exp(1j*kvals.T*k), [2, 3], [3, 2])
    Fk = np.dot(omega_phi_nu, z)
    Fk0 = Fk[map3[ :, 0], map3[ :, 1], :]
    Fk1 = F0_expanded[map3[ :, 2], map3[ :, 3], :]
    Fk2 = Fk[map3[ :, 4], map3[ :, 5], :]
    S3_k_neigh = np.sum(Fk0 * Fk1 * Fk2, axis=1)
    
    omega_phi_nu0 = omega_phi_nu[map3[ :, 0], map3[ :, 1], :, :]
    omega_phi_nu1 = omega0_expanded[map3[ :, 2], map3[ :, 3], :, :]
    omega_phi_nu2 = omega_phi_nu[map3[ :, 4], map3[ :, 5], :, :]
    
    Fk0 = np.repeat(Fk0[ :, :, np.newaxis], len(z), axis=2)
    Fk1 = np.repeat(Fk1[ :, :, np.newaxis], len(z), axis=2)
    Fk2 = np.repeat(Fk2[ :, :, np.newaxis], len(z), axis=2)
    gS3_k_neigh = np.sum(Fk0 * Fk1 * omega_phi_nu2 + Fk0 * Fk2 * omega_phi_nu1 + Fk1 * Fk2 * omega_phi_nu0, axis=1)
    
    # %% Rotationally-averaged third-order autocorrelation with two neighbors
    Fk0 = np.atleast_2d(F0[map3[ :, 0], map3[ :, 1]])
    Fk1 = np.atleast_2d(F0[map3[ :, 2], map3[ :, 3]])
    Fk2 = np.atleast_2d(F0[map3[ :, 4], map3[ :, 5]])
    S3_k_triplets = np.squeeze(Fk0 * Fk1 * Fk2)
    
    omega_phi_nu0 = omega0[map3[ :, 0], map3[ :, 1], :]
    omega_phi_nu1 = omega0[map3[ :, 2], map3[ :, 3], :]
    omega_phi_nu2 = omega0[map3[ :, 4], map3[ :, 5], :]
    
    Fk0 = np.repeat(Fk0[ :, :, np.newaxis], len(z), axis=2)
    Fk1 = np.repeat(Fk1[ :, :, np.newaxis], len(z), axis=2)
    Fk2 = np.repeat(Fk2[ :, :, np.newaxis], len(z), axis=2)
    gS3_k_triplets = np.squeeze(Fk0 * Fk1 * omega_phi_nu2 + Fk0 * Fk2 * omega_phi_nu1 + Fk1 * Fk2 * omega_phi_nu0)
    return S3_k, gS3_k, S3_k_neigh, gS3_k_neigh, S3_k_triplets, gS3_k_triplets


def calc_gpsf(L, S3_x_neigh):
    gpsf = sp.lil_matrix((L**4, (4*L-3)**2))
    for i1 in range(L):
        for j1 in range(L):
            for i2 in range(L):
                for j2 in range(L):
                    shift1y = j1
                    shift1x = i1
                    shift2y = j2
                    shift2x = i2
                            
                    row = np.ravel_multi_index([i1, j1, i2, j2], (L, L, L, L))
                    
                    for j in range(max(shift1y, shift2y) - (L-1), L + min(shift1y, shift2y)):
                        for i in range(max(shift1x, shift2x) - (L-1), L + min(shift1x, shift2x)):
                            if not (np.abs(i) < L and np.abs(j) < L):
                                gpsf[row, np.ravel_multi_index([i + 2*L-2, j + 2*L-2], (4*L-3, 4*L-3))] += S3_x_neigh[(shift2x-shift1x)%(2*L-1), (shift2y-shift1y)%(2*L-1), (i-shift1x)%(2*L-1), (j-shift1y)%(2*L-1)]
                    
                    for j in range(shift1y - (L-1), L + shift1y - shift2y):
                        for i in range(shift1x - (L-1), L + shift1x - shift2x):
                            if not (np.abs(i) < L and np.abs(j) < L):
                                gpsf[row, np.ravel_multi_index([i + 2*L-2, j + 2*L-2], (4*L-3, 4*L-3))] += S3_x_neigh[shift2x%(2*L-1), shift2y%(2*L-1), (shift1x-i)%(2*L-1), (shift1y-j)%(2*L-1)]
                    
                    for j in range(shift2y - (L-1), L + shift2y - shift1y):
                        for i in range(shift2x - (L-1), L + shift2x - shift1x):
                            if not (np.abs(i) < L and np.abs(j) < L):
                                gpsf[row, np.ravel_multi_index([i + 2*L-2, j + 2*L-2], (4*L-3, 4*L-3))] += S3_x_neigh[shift1x%(2*L-1), shift1y%(2*L-1), (shift2x-i)%(2*L-1), (shift2y-j)%(2*L-1)]
    return sp.csr_matrix(gpsf)