# -*- coding: utf-8 -*-
"""
Created on Sat Oct 12 21:14:05 2019

@author: Shay Kreymer
"""
import numpy as np

def calc_estimation_error(X, X_estimated):
    """Calculate relative estimation error between the ground truth and the estimation.

    Keyword arguments:
    X -- the ground truth image
    X_estimated -- the estimated image
    """
    return np.linalg.norm(X - X_estimated, ord="fro")/np.linalg.norm(X)
