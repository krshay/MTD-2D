# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 22:08:53 2019

@author: kreym
"""
import itertools

import numpy as np

def calclist2(L):
    """Calculate list of shifts for second order moment.

    Keyword arguments:
    L - size of image
    """
    list2 = list(itertools.product(list(range(L)), repeat=2))
    list2.remove((0, 0))
    return list2

def calclist3(L):
    """Calculate list of shifts for third order moment.

    Keyword arguments:
    L - size of image
    """
    list2 = calclist2(L)
    list3 = list(itertools.combinations_with_replacement(list2, 2))
    for element in list3:
        if element[0] == element[1]:
            list3.remove(element)
    return list3

def calclist2_all_ver(L):
    list2 = list(itertools.product(list(range(-L+1,L)), repeat=2))
    list2.remove((0, 0))
    return list2

def calclist2_all(L):
    lst2 = calclist2(L)
    list2 = []
    for shift in lst2:
        shift1y = shift[0]
        shift1x = shift[1]
        # %% 2.
        for j in range(shift1y-(L-1), L + shift1y):
            for i in range(shift1x-(L-1), L + shift1x):
                if not (np.abs(i) < L and np.abs(j) < L):
                    list2.append((j-shift1y, i-shift1x))
    return list2

def calclist3_all_ver(L):
    """Calculate list of shifts for third order moment.

    Keyword arguments:
    L - size of image
    """
    list2 = calclist2_all_ver(L)
    list3 = list(itertools.combinations_with_replacement(list2, 2))
    for element in list3:
        if element[0] == element[1]:
            list3.remove(element)
    return list3

def calclist3_all(L, triplets=False):
    lst3 = calclist3(L)
    list3 = []
    for shift in lst3:
        shift1y = shift[0][0]
        shift1x = shift[0][1]
        shift2y = shift[1][0]
        shift2x = shift[1][1]
        # %% 2. 
        for j in range(max(shift1y, shift2y) - (L-1), L + min(shift1y, shift2y)):
            for i in range(max(shift1x, shift2x) - (L-1), L + min(shift1x, shift2x)):
                if not (np.abs(i) < L and np.abs(j) < L):
                    list3.append(((shift2y-shift1y, shift2x-shift1x), (j-shift1y, i-shift1x)))

        # %% 3. 
        for j in range(shift1y - (L-1), L + shift1y - shift2y):
            for i in range(shift1x - (L-1), L + shift1x - shift2x):
                if not (np.abs(i) < L and np.abs(j) < L):
                    list3.append(((j-shift1y, i-shift1x), (j+shift2y-shift1y, i+shift2x-shift1x)))

        # %% 4. 
        for j in range(shift2y - (L-1), L + shift2y - shift1y):
            for i in range(shift2x - (L-1), L + shift2x - shift1x):
                if not (np.abs(i) < L and np.abs(j) < L):
                    list3.append(((j-shift2y, i-shift2x), (j+shift1y-shift2y, i+shift1x-shift2x)))
                    
        if triplets:
            # %% 5.                 
            for j1 in range(shift1y - (L-1), L + shift1y):
                for i1 in range(shift1x - (L-1), L + shift1x):
                    if not (np.abs(i1) < L and np.abs(j1) < L):
                        for j2 in range(max(j1, shift1y) - (L-1) - shift2y, L + min(j1, shift1y) - shift2y):
                            for i2 in range(max(i1, shift1x) - (L-1) - shift2x, L + min(i1, shift1x) - shift2x):
                                if not (np.abs(i2) < L and np.abs(j2) < L):
                                    if not (np.abs(i2 - i1) < L and np.abs(j2 - j1) < L):
                                        list3.append(((j1 - shift1y, i1 - shift1x), (j2 + shift2y - shift1y, i2 + shift2x - shift1x)))

            # $$ 6.
            for j1 in range(shift2y - (L-1), L + shift2y):
                for i1 in range(shift2x - (L-1), L + shift2x):
                    if not (np.abs(i1) < L and np.abs(j1) < L):
                        for j2 in range(max(j1, shift2y) - (L-1) - shift1y, L + min(j1, shift2y) - shift1y):
                            for i2 in range(max(i1, shift2x) - (L-1) - shift1x, L + min(i1, shift2x) - shift1x):
                                if not (np.abs(i2) < L and np.abs(j2) < L):
                                    if not (np.abs(i2 - i1) < L and np.abs(j2 - j1) < L):
                                        list3.append(((j1 - shift2y, i1 - shift2x), (j2 + shift1y - shift2y, i2 + shift1x - shift2x)))

    return list(dict.fromkeys(list3))
