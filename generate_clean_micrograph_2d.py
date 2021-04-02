# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 16:40:17 2019

@author: kreym
"""
import numpy as np
from fb_funcs import rot_img_freqT

def generate_clean_micrograph_2d(X, W, N, m, p=np.array([1])):
    """Form an N*N matrix containing matrices from X with probability p.

    Keyword arguments:
    X -- a tuple of required images
    W -- the width and height of the images to fulfill the separation condition (1.2)
    N -- the width and height of the required micrograph
    m -- wanted number of images to place
    p -- choosing probability of images in X (default np.array([1]))
    """
    m = round(m)
    mask = np.zeros((N, N))
    # The locations table records the chosen signal locations.
    locations = [[] for i in range(m)]
    # This counter records how many signals we successfully placed.
    placed = 0
    placed_list = [0 for i in range(len(X))]
    max_trials = 4*m
    L = np.shape(X[0])[0]
    Y = np.zeros((N, N))
    for _ in range(max_trials):
        # Pick a candidate location for the upper-left corner of the signal
        candidate = np.random.randint(N-W, size=2)
        # Check if there is enough room, taking the separation rule into
        # account. That is, a square of size WxW with upper-left corner
        # specified by the candidate must be entirely free.
        if (mask[candidate[0]:candidate[0]+W, candidate[1]:candidate[1]+W] == 0).all():
            # Record the successful candidate
            locations[placed] = candidate
            placed = placed + 1
            # Mark the area as reserved
            mask[candidate[0]:candidate[0]+W, candidate[1]:candidate[1]+W] = 1
            index_rand = np.random.choice(len(X), p=p)
            placed_list[index_rand] = placed_list[index_rand] + 1
            Y[candidate[0] : candidate[0] + L, candidate[1] : candidate[1] + L] = X[index_rand]
            # Stop if we placed sufficiently many signals successfully.
            if placed >= m:
                break
    locations = locations[0:placed]
    #    Y = zeros(N, N,'gpuArray');
    return Y, placed_list, locations

def generate_clean_micrograph_2d_one_neighbor(X, W, N, m, p=np.array([1])):
    m = round(m)
    mask = np.zeros((N, N))
    # The locations table records the chosen signal locations.
    locations = [[] for i in range(m)]
    # This counter records how many signals we successfully placed.
    placed = 0
    placed_list = [0 for i in range(len(X))]
    max_trials = 4*m
    L = np.shape(X[0])[0]
    Y = np.zeros((N, N))
    for _ in range(max_trials):
        # Pick a candidate location for the upper-left corner of the signal
        candidate = np.random.randint(N-W, size=2)
        # Check if there is enough room, taking the separation rule into
        # account. That is, a square of size WxW with upper-left corner
        # specified by the candidate must be entirely free.
        if not((mask[candidate[0]:candidate[0]+2*W-1, candidate[1]:candidate[1]+2*W-1] == 1).any()):
            # Record the successful candidate
            locations[placed] = candidate
            placed = placed + 1
            # Mark the area as reserved
            mask[candidate[0]:candidate[0]+W, candidate[1]:candidate[1]+W] = 1/2
            mask[candidate[0]:candidate[0]+2*W-1, candidate[1]:candidate[1]+2*W-1] += 1/2
            index_rand = np.random.choice(len(X), p=p)
            placed_list[index_rand] = placed_list[index_rand] + 1
            Y[candidate[0] : candidate[0] + L, candidate[1] : candidate[1] + L] = X[index_rand]
            # Stop if we placed sufficiently many signals successfully.
            if placed >= m:
                break
    locations = locations[0:placed]
    #    Y = zeros(N, N,'gpuArray');
    return Y, placed_list, locations

def generate_clean_micrograph_2d_one_neighbor_rots(c, kvals, Bk, W, L, N, m, T, p=np.array([1])):
    m = round(m)
    thetas = np.linspace(0, 2*np.pi, m)
    thetas = np.random.uniform(low=0, high=2*np.pi, size=(m, ))
    mask = np.zeros((N, N))
    # The locations table records the chosen signal locations.
    locations = [[] for i in range(m)]
    # This counter records how many signals we successfully placed.
    placed = 0
    placed_list = [0 for i in range(1)]
    max_trials = 5*m
    Y = np.zeros((N, N))
    for _ in range(max_trials):
        # Pick a candidate location for the upper-left corner of the signal
        candidate = np.random.randint(N-W, size=2)
        # Check if there is enough room, taking the separation rule into
        # account. That is, a square of size WxW with upper-left corner
        # specified by the candidate must be entirely free.
        if not((mask[candidate[0]:candidate[0]+2*W-1, candidate[1]:candidate[1]+2*W-1] == 1).any()):
            # Record the successful candidate
            locations[placed] = candidate
            # Mark the area as reserved
            mask[candidate[0]:candidate[0]+W, candidate[1]:candidate[1]+W] = 1/2
            mask[candidate[0]:candidate[0]+2*W-1, candidate[1]:candidate[1]+2*W-1] += 1/2
            index_rand = np.random.choice(1, p=p)
            placed_list[index_rand] = placed_list[index_rand] + 1
            theta = thetas[placed]
            placed = placed + 1
            X_theta = rot_img_freqT(theta, c, kvals, Bk, L, T)
            Y[candidate[0] : candidate[0] + L, candidate[1] : candidate[1] + L] = X_theta
            # Stop if we placed sufficiently many signals successfully.
            if placed >= m:
                break
    locations = locations[0:placed]
    #    Y = zeros(N, N,'gpuArray');
    return Y, placed_list, locations

def generate_clean_micrograph_2d_rots(c, kvals, Bk, W, L, N, m, T, p=np.array([1]), seed=None):
    """Form an N*N matrix containing matrices from X with probability p.

    Keyword arguments:
    X -- a tuple of required images
    W -- the width and height of the images to fulfill the separation condition (1.2)
    N -- the width and height of the required micrograph
    m -- wanted number of images to place
    p -- choosing probability of images in X (default np.array([1]))
    """
    if seed != None:
        np.random.seed(seed)
    m = round(m)
    thetas = np.linspace(0, 2*np.pi, m)
    thetas = np.random.uniform(low=0, high=2*np.pi, size=(m, ))
    mask = np.zeros((N, N))
    # The locations table records the chosen signal locations.
    locations = [[] for i in range(m)]
    # This counter records how many signals we successfully placed.
    placed = 0
    placed_list = [0 for i in range(1)]
    max_trials = 5*m
    Y = np.zeros((N, N))
    for _ in range(max_trials):
        # Pick a candidate location for the upper-left corner of the signal
        candidate = np.random.randint(N-W, size=2)
        # Check if there is enough room, taking the separation rule into
        # account. That is, a square of size WxW with upper-left corner
        # specified by the candidate must be entirely free.
        if not((mask[candidate[0]:candidate[0]+2*W-1, candidate[1]:candidate[1]+2*W-1] == 1).any()):
            # Record the successful candidate
            locations[placed] = candidate
            # Mark the area as reserved
            mask[candidate[0]:candidate[0]+W, candidate[1]:candidate[1]+W] = 1
            index_rand = np.random.choice(1, p=p)
            placed_list[index_rand] = placed_list[index_rand] + 1
            theta = thetas[placed]
            placed = placed + 1
            X_theta = rot_img_freqT(theta, c, kvals, Bk, L, T)
            Y[candidate[0] : candidate[0] + L, candidate[1] : candidate[1] + L] = X_theta
            # Stop if we placed sufficiently many signals successfully.
            if placed >= m:
                break
    locations = locations[0:placed]
    #    Y = zeros(N, N,'gpuArray');
    return Y, placed_list, locations
