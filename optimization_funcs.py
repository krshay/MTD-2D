# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 13:56:03 2020

@author: kreym
"""

import numpy as np
from scipy.optimize import minimize
import multiprocessing as mp
from funcs_calc_lists import calclist2_all, calclist3_all
import c_g_funcs
import psf_functions_2d as psf_funcs

import tensorflow as tf

def optimize_2d_known_psf(initial_guesses, M1_Y, M2_Y, M3_Y, list2, list3, full_psf, triplets, L, K, gtol=1e-10):
    # Optimization assuming known psf
    list2_all = calclist2_all(L)
    list3_all = calclist3_all(L)
    # list3_all_reversed = [t[::-1] for t in list3_all]
    return minimize(fun=c_g_funcs.cost_grad_fun_2d_known_psf, x0=initial_guesses, method='BFGS', jac=True, options={'disp': False, 'gtol': gtol}, args = ([M1_Y, M2_Y, M3_Y, list2, list3, 0, 0], full_psf, triplets, L, K, list2_all, list3_all))

def optimize_2d_known_psf_parallel(initial_guesses, M1_Y, M2_Y, M3_Y, list2, list3, full_psf, triplets, L, K, gtol=1e-10):
    # Optimization assuming known psf - using parallel processing
    list2_all = calclist2_all(L)
    list3_all = calclist3_all(L)
    list3_all_reversed = [t[::-1] for t in list3_all]
    pool = mp.Pool()
    return minimize(fun=c_g_funcs.cost_grad_fun_2d_known_psf, x0=initial_guesses, method='BFGS', jac=True, options={'disp': False, 'gtol': gtol}, args = ([M1_Y, M2_Y, M3_Y, list2, list3, 0, 0], full_psf, triplets, L, K, list2_all, list3_all, list3_all_reversed, pool))

def optimize_2d_full_psf_estimation(initial_guesses, M1_Y, M2_Y, M3_Y, list2, list3, L, K, psf0=None, gtol=1e-10):
    # Optimization + estimating full (2-D) psf
    if psf0 is None:
        psf0, _ = psf_funcs.psf_initial_guess(L, initial_guesses[0])
    initial_guesses = np.concatenate((initial_guesses, psf0.flatten()))
    list2_all = calclist2_all(L)
    list3_all = calclist3_all(L)
    # list3_all_reversed = [t[::-1] for t in list3_all]
    return minimize(fun=c_g_funcs.cost_grad_fun_2d_full_psf, x0=initial_guesses, method='BFGS', jac=True, options={'disp': False, 'gtol': gtol}, args = ([M1_Y, M2_Y, M3_Y, list2, list3, 0, 0], L, K, list2_all, list3_all))

def optimize_2d_full_psf_no_estimation(initial_guesses, M1_Y, M2_Y, M3_Y, list2, list3, triplets, L, K, full_psf=None, gtol=1e-10):
    # Optimization using some initial psf, without estimating psf
    if full_psf is None:
        full_psf, _ = psf_funcs.psf_initial_guess(L, initial_guesses[0])
    list2_all = calclist2_all(L)
    list3_all = calclist3_all(L)
    # list3_all_reversed = [t[::-1] for t in list3_all]
    return minimize(fun=c_g_funcs.cost_grad_fun_2d_known_psf, x0=initial_guesses, method='BFGS', jac=True, options={'disp': False, 'gtol': gtol}, args = ([M1_Y, M2_Y, M3_Y, list2, list3, 0, 0], full_psf, triplets, L, K, list2_all, list3_all))

def optimize_2d_psf_expansion(initial_guesses, M1_Y, M2_Y, M3_Y, list2, list3, L, K, coeffs0=None, gtol=1e-10):
    # Optimization using Fourier-Bessel expansion of the psf
    if coeffs0 is None:
        _, locations = psf_funcs.psf_initial_guess(L, initial_guesses[0])
        num_of_coeffs = L
        coeffs0 = psf_funcs.coeffs_initial_guess(L, locations, num_of_coeffs)
    initial_guesses = np.concatenate((initial_guesses, coeffs0.flatten()))
    list2_all = calclist2_all(L)
    list3_all = calclist3_all(L)
    # list3_all_reversed = [t[::-1] for t in list3_all]
    return minimize(fun=c_g_funcs.cost_grad_fun_2d_psf_expansion, x0=initial_guesses, method='BFGS', jac=True, options={'disp': False, 'gtol': gtol}, args = ([M1_Y, M2_Y, M3_Y, list2, list3, 0, 0], L, K, list2_all, list3_all))

def optimize_2d_known_psf_Adam(initial_guesses, M1_Y, M2_Y, M3_Y, list2, list3, full_psf, triplets, L, K, gtol=1e-10):
    # Optimization assuming known psf
    optizer = tf.compat.v1.train.AdamOptimizer()
    initial_guesses = tf.Variable(initial_value=tf.convert_to_tensor(initial_guesses))
    list2_all = calclist2_all(L)
    list3_all = calclist3_all(L)
    # list3_all_reversed = [t[::-1] for t in list3_all]
    
    return optizer.minimize(tf.convert_to_tensor(c_g_funcs.cost_fun_2d_known_psf(initial_guesses, [M1_Y, M2_Y, M3_Y, list2, list3, 0, 0], full_psf, triplets, L, K, list2_all, list3_all)), var_list=initial_guesses)#([M1_Y, M2_Y, M3_Y, list2, list3, 0, 0], full_psf, triplets, L, K, list2_all, list3_all))
