# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 19:43:46 2019

@author: kreym
"""
import numpy as np
import scipy.spatial as spatial
import scipy.integrate as integrate
import scipy.special as special

from generate_clean_micrograph_2d import generate_clean_micrograph_2d
# from scipy.optimize import least_squares as ls

def numeric_psf_2d(locations, L):
    M = len(locations)
    r2_max = round(2*(2*L-2)**2)
    psf = np.zeros(r2_max+1)
    locations_tree = spatial.cKDTree(locations)
    vectors = {}
    for i in range(0, r2_max+1):
        vectors[i] = []
    for loc in locations:
        close_locs = [locations[j] for j in locations_tree.query_ball_point(loc, np.sqrt(r2_max))]
        close_locs = [close_loc for close_loc in close_locs if not ((np.abs(loc[0] - close_loc[0]) >= 2*L-1)
                                                                    or (np.abs(loc[1] - close_loc[1]) >= 2*L-1))]
        
        for vec in (loc - close_locs):
            vec = list(vec)
            norm_vec_squared = (np.round((np.linalg.norm(vec))**2)).astype(int)
            if vec not in vectors[norm_vec_squared]:
                vectors[norm_vec_squared].append(vec)
                    
        dists = np.round(np.linalg.norm(loc - close_locs, axis=1)**2).astype(int)
        for dist in dists:
            if dist != 0:
                psf[dist] += (1)/(M)
    for key in vectors:
        vectors[key] = len(vectors[key])
    numofdistances = list(vectors.values())
    for i in range(r2_max+1):
        if numofdistances[i] != 0:
            psf[i] = psf[i] / numofdistances[i]
            
    return psf, numofdistances

def fourier_bessel_coeffs_ls(psf_vals, N, b, L):
    roots = special.jn_zeros(0,N)
    r = np.sqrt(np.arange(0, b**2))
    psf = np.zeros(len(r))
    psf[0:len(psf_vals)] = psf_vals
    nonzeros = np.nonzero(psf)
    r = r[nonzeros]
    r = np.concatenate((np.arange(L), r))
    psf = psf[nonzeros]
    psf = np.concatenate((np.zeros(L), psf))
    R, Roots = np.meshgrid(r, roots)
    a = special.j0(((R * Roots).T)/b)
    coeffs, residuals, rank, s = np.linalg.lstsq(a, psf, rcond=None)
    return coeffs
    
def fourier_bessel_expansion(coeffs, b, r):
    # Calculates the value of the function in r using the Fourier-Bessel coefficients
    N = len(coeffs)
    roots = special.jn_zeros(0, N)
    Js = [special.j0(roots[n]*r/b) for n in range(N)]
    return np.array(coeffs).T @ np.array(Js)

def full_psf_2d(locations, L):
    M = len(locations)
    r_max = np.sqrt(2)*(2*L-2)
    psf = np.zeros((4*L-3, 4*L-3))
    locations_tree = spatial.cKDTree(locations)
    
    for loc in locations:
        close_locs = [locations[j] for j in locations_tree.query_ball_point(loc, r_max)]
        close_locs = [close_loc for close_loc in close_locs if not ((np.abs(loc[0] - close_loc[0]) >= 2*L-1)
                                                                    or (np.abs(loc[1] - close_loc[1]) >= 2*L-1))]
        for close_loc in close_locs:
            dif = np.array(loc) - np.array(close_loc)
            psf[dif[0]+2*L-2, dif[1]+2*L-2] += 1/(M)
    psf[2*L-2, 2*L-2] = 0

    return psf

def numeric_psf2full(numeric_psf, L):
    psf = np.zeros((4*L-3, 4*L-3))
    
    for j in range(4*L-3):
        for i in range(4*L-3):
            if not (np.abs(i-(2*L-2)) < L and np.abs(j-(2*L-2)) < L):
                psf[j, i] = evaluate_psf(numeric_psf, np.sqrt((i-(2*L-2))**2 + (j-(2*L-2))**2))  
    return psf

def calc_psf_from_coeffs(coeffs, L):
    psf = np.zeros((4*L-3, 4*L-3))
    b = np.sqrt(2)*(2*L-1)
    for j in range(4*L-3):
        for i in range(4*L-3):
            if not (np.abs(i-(2*L-2)) < L and np.abs(j-(2*L-2)) < L):
                psf[j, i] = fourier_bessel_expansion(coeffs,b, np.sqrt((i-(2*L-2))**2 + (j-(2*L-2))**2))
    return psf

def psf_initial_guess(L, gamma):
    gamma = float(gamma)
    X0 = np.random.rand(L, L)
    Xs = (X0, )
    N = L*100
    W = L
    Y, s_abs, locations = generate_clean_micrograph_2d(Xs, W, N, gamma*(N/W)**2)#, p=np.array([1/2, 1/2]))
    return full_psf_2d(locations, L), locations

def coeffs_initial_guess(L, locations, num_of_coeffs):
    numeric_psf, _ = numeric_psf_2d(locations, L)

    b = np.sqrt(2)*(2*L-1)

    return fourier_bessel_coeffs_ls(numeric_psf, num_of_coeffs, b, L)

def evaluate_psf(psf, r):
    return psf[round(r**2).astype(int)]

def evaluate_psf_full(psf, r, L):
    return psf[r[0] + 2*L-2, r[1] + 2*L-2]
