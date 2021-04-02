# -*- coding: utf-8 -*-
"""
Created on Sat Aug 15 11:41:38 2020

@author: kreym
"""

import numpy as np
from scipy.linalg import circulant, dft

def generate_signal(beta, L):
    M = L**2
    sigma_f = np.zeros((M, ))
    sigma_f[0] = 1; # normalized DC
    if M % 2 == 0:
        sigma_f[1:M//2+1] = 1/(np.arange(1, M/2+1)**beta)
        sigma_f[M//2+1:] = sigma_f[1:M//2+1][::-1]
    else:
        sigma_f[1:(M+1)//2] = 1/(np.arange(1, (M+1)/2)**beta)
        sigma_f[(M+1)//2:] = sigma_f[1:(M+1)//2][::-1]
    
    F = dft(L)
    SIGMA1 = circulant(np.real(np.fft.ifft(sigma_f))) # signal's covariance matrix
    SIGMA2 = np.conjugate(np.kron(F, F)).T * np.diag(sigma_f) * np.kron(F, F)# signal's covariance matrix
    x = np.random.multivariate_normal(np.zeros((M, )), SIGMA1)
    x = x/np.linalg.norm(x)
    return np.reshape(x, (L, L))