# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 19:43:46 2019

@author: kreym
"""
import numpy as np
import scipy.spatial as spatial

def full_tsf_2d(locations, L):
    M = len(locations)
    r_max = np.sqrt(2)*(2*L-2)
    tsf = np.zeros((4*L-3, 4*L-3, 4*L-3, 4*L-3))
    locations_tree = spatial.cKDTree(locations)
    
    for loc in locations:
        close_locs = [locations[j] for j in locations_tree.query_ball_point(loc, r_max)]
        close_locs = [close_loc for close_loc in close_locs if not ((np.abs(loc[0] - close_loc[0]) >= 2*L-1)
                                                                    or (np.abs(loc[1] - close_loc[1]) >= 2*L-1)
                                                                    or (loc[0] - close_loc[0] == 0 and loc[1] - close_loc[1] == 0))]
        
        for close_loc1 in close_locs:
            close_locs_reduced = [close_loc for close_loc in close_locs if close_loc[0] != close_loc1[0] and close_loc[1] != close_loc1[1]]
            for close_loc2 in close_locs_reduced:
                dif1 = np.array(loc) - np.array(close_loc1)
                dif2 = np.array(loc) - np.array(close_loc2)
                tsf[dif1[0]+2*L-2, dif1[1]+2*L-2, dif2[0]+2*L-2, dif2[1]+2*L-2] += 1/(M**2)
    tsf[2*L-2, 2*L-2, :, :] = 0
    return tsf

