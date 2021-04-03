# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 20:00:31 2020

@author: kreym
"""

import numpy as np
import scipy.special as spl
import scipy.sparse as sp
from Utils.calc_estimation_error import calc_estimation_error

def expand_fb(img, ne):
    '''
    Inputs:
        img: 2D image to be expanded
        ne: number of expansion coefficients
    
    Outputs:
        B: matrix that maps from Utils.the expansion coefficients to
           the approximated image
        z: expansion coefficients in complex format
        roots: roots of the Bessel functions
        kvals: order of the Bessel functions
        nu: modified number of expansion coefficients
    '''
    n1 = np.shape(img)[0]
    R = n1 // 2
    r_limit = np.pi*R
    tol = 1e-10

    # generate a table for the roots of Bessel functions
    max_num_root = 0
    bessel_roots = spl.jn_zeros(0, ne)
    for i in range(ne):
        if (bessel_roots[i] - r_limit > tol):
            break
        max_num_root += 1

    kmax = 1
    vmax = bessel_roots[max_num_root-1]
    while (1):
        if (spl.jn_zeros(kmax, 1)[0] < vmax):
            kmax += 1
        else:
            break

    kt = np.zeros((kmax, max_num_root), dtype=np.int)
    rt = np.zeros((kmax, max_num_root))
    for k in range(kmax):
        kt[k, :] = k
        rt[k, :] = spl.jn_zeros(k, max_num_root)

    # sort the Bessel roots in ascending order
    ktv = kt.flatten()
    rtv = rt.flatten()
    idx = np.argsort(rtv)
    ks = ktv[idx]
    rs = rtv[idx]

    # count number of basis functions
    nu = 0
    for i in range(len(ks)):
        if (ks[i] == 0):
            nu += 1
        else:
            nu += 2
        if (nu >= ne):
            break

    # array that indicates cosine (0) or sine (1)
    td = np.zeros(nu, dtype=np.int)
    kd = np.zeros(nu, dtype=np.int)
    rd = np.zeros(nu)

    count = 0
    for i in range(len(ks)):
        kd[count] = ks[i]
        rd[count] = rs[i]
        td[count] = 0
        count += 1
        if (ks[i] > 0):
            kd[count] = ks[i]
            rd[count] = rs[i]
            td[count] = 1
            count += 1
        if (count == nu):
            break

    roots = np.copy(rd)
    kvals = np.zeros_like(kd)
    for i in range(nu):
        kvals[i] = kd[i]*(1 - 2*td[i])

    # consider pixels within the circular support
    x = np.arange(-R, R+1)
    y = np.arange(-R, R+1)
    xv, yv = np.meshgrid(x, y)
    radi = np.sqrt(xv**2 + yv**2) / (R+1)
    theta = np.arctan2(yv, xv)
    theta = theta[radi < 1]
    num_pix = np.sum([radi < 1])

    # evaluate basis functions within the support
    E = np.zeros((num_pix, nu))
    for i in range(nu):
        rvals = rd[i] * radi[radi < 1]
        if (td[i] == 0):
            E[:, i] = spl.jv(kd[i], rvals) * np.cos(kd[i]*theta)
        else:
            E[:, i] = spl.jv(kd[i], rvals) * np.sin(kd[i]*theta)

    # expansion coefficients in real representation
    bvec = img[radi < 1]
    sol = np.linalg.lstsq(E, bvec, rcond=None)
    x = sol[0]

    Bp = np.zeros((num_pix, nu), dtype=complex)
    z = np.zeros(nu, dtype=complex)
    for i in range(nu):
        if (kd[i] == 0):
            Bp[:, i] = E[:, i]
            z[i] = x[i]
            continue
        if (td[i] == 0):
            Bp[:, i] = E[:, i] + 1j*E[:, i+1]
            z[i] = (x[i] - 1j*x[i+1])/2
        else:
            Bp[:, i] = E[:, i-1] - 1j*E[:, i]
            z[i] = (x[i-1] + 1j*x[i])/2

    B = np.zeros((n1**2, nu), dtype=complex)
    I = np.zeros((n1, n1), dtype=complex)
    for i in range(nu):
        I[radi < 1] = Bp[:, i]
        B[:, i] = I.flatten()

    return (B, z, roots, kvals, nu)

def rot_img(theta, z, kvals, B):
    '''
    Rotate image by angle theta (in radians)
    '''
    rot_z = np.zeros(np.shape(z), dtype=np.complex_)
    for i in range(len(z)):
        rot_z[i] = z[i]*np.exp(1j*theta*kvals[i])

    rot_img = np.real(B@rot_z)
    n1 = np.int(np.sqrt(len(rot_img)))
    rot_img = np.reshape(rot_img, (n1, n1))

    return rot_img

def rot_img_freq(theta, z, kvals, Bk, L):
    return np.real(np.fft.ifft2((Bk*np.exp(1j*kvals*theta)@z)))[L//2:-(L//2), L//2:-(L//2)]

def rot_img_freqT(theta, c, kvals, Bk, L, T):
    return np.real(np.fft.ifft2((Bk*np.exp(1j*kvals*theta)@(T.H@c))))[L//2:-(L//2), L//2:-(L//2)]

def min_err_rots(z, z_est, kvals, B, L):
    X = np.reshape(np.real(B @ z), (L, L))
    thetas = np.linspace(0, 2*np.pi, 360)
    errs = np.zeros_like(thetas)
    for t in range(len(thetas)):
        X_rot = rot_img(thetas[t], z_est, kvals, B)
        errs[t] = calc_estimation_error(X, X_rot)
    return np.min(errs), thetas[np.argmin(errs)]

def min_err_coeffs(z, z_est, kvals):
    thetas = np.linspace(0, 2*np.pi, 3600)
    z_est_rot = np.zeros(np.shape(z), dtype=np.complex_)
    errs = np.zeros_like(thetas)
    for t in range(len(thetas)):
        z_est_rot = z_est*np.exp(1j*thetas[t]*kvals)
        errs[t] = np.linalg.norm(z-z_est_rot, ord=2)/np.linalg.norm(z, ord=2)
    return np.min(errs), thetas[np.argmin(errs)]

def calcT(nu, kvals):
    v = np.zeros(2*nu).astype(np.complex)
    iv = np.zeros(2*nu).astype(np.int)
    jv = np.zeros(2*nu).astype(np.int)
    jj = 0
    for ii in range(nu):
        if kvals[ii] == 0:
            v[jj] = 1
            iv[jj] = ii
            jv[jj] = ii
            jj += 1
        if kvals[ii] > 0:
            v[jj] = 1/np.sqrt(2)
            iv[jj] = ii
            jv[jj] = ii
            jj += 1
            v[jj] = 1/np.sqrt(2)
            iv[jj] = ii
            jv[jj] = ii + 1
            jj += 1
        if kvals[ii] < 0:
            v[jj] = 1j/np.sqrt(2)
            iv[jj] = ii 
            jv[jj] = ii - 1
            jj += 1
            v[jj] = -1j/np.sqrt(2)
            iv[jj] = ii
            jv[jj] = ii
            jj += 1
    v = v[0:jj] 
    iv = iv[0:jj]
    jv = jv[0:jj]
    T = sp.csr_matrix((v,(iv,jv)),shape=(nu, nu))  
    return T
    