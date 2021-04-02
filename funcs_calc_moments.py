# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 22:10:47 2019

@author: kreym
"""
import numpy as np
def calcM2_2d(A, list2):
    """Calculate second order 2-d autocorrelation of A for shifts in list2.

    Keyword arguments:
    A -- the 2-d signal
    list2 -- a list containing shifts
    """
    dim1, dim2 = np.shape(A)
    n2 = np.shape(list2)[0]
    M2 = np.zeros(n2)

    for k in range(n2):
        shift1 = list2[k]
        shift1y = -shift1[0]
        shift1x = -shift1[1]

        valsy1 = [0, -shift1y]
        valsx1 = [0, -shift1x]

        rangey = list(range(max(valsy1), dim1+min(valsy1)))
        rangey1 = [x + shift1y for x in rangey]
        rangex = list(range(max(valsx1), dim2+min(valsx1)))
        rangex1 = [x + shift1x for x in rangex]

        M2[k] = np.sum(A[min(rangey):max(rangey)+1, min(rangex):max(rangex)+1] * A[min(rangey1):max(rangey1)+1, min(rangex1):max(rangex1)+1])/np.size(A)

    return M2

#    sA = np.shape(A);
#    n2 = np.shape(list2)[0];
#    M2 = np.zeros(n2);

#    for k in range(n2):
#        shift1 = list2[k];
#        A_padded = np.pad(A,((shift1[0], 0), (shift1[1], 0)), mode='constant');
#        A_padded = A_padded[0:sA[0], 0:sA[1]];
#        M2[k] = np.sum(np.multiply(A, A_padded))/np.size(A);

#    return M2;

def calcM3_2d(A, list3):
    """Calculate third order 2-d autocorrelation of A for shifts in list3.

    Keyword arguments:
    A -- the 2-d signal
    list3 -- a list containing shifts
    """
    dim1, dim2 = np.shape(A)
    n3 = np.shape(list3)[0]
    M3 = np.zeros(n3)

    for k in range(n3):
        shift1 = list3[k][0]
        shift1y = -shift1[0]
        shift1x = -shift1[1]

        shift2 = list3[k][1]
        shift2y = -shift2[0]
        shift2x = -shift2[1]

        valsy1 = [0, -shift1y, -shift2y]
        valsx1 = [0, -shift1x, -shift2x]

        rangey = list(range(max(valsy1), dim1+min(valsy1)))
        rangey1 = [x + shift1y for x in rangey]
        rangey2 = [x + shift2y for x in rangey]

        rangex = list(range(max(valsx1), dim2+min(valsx1)))
        rangex1 = [x + shift1x for x in rangex]
        rangex2 = [x + shift2x for x in rangex]

        if rangey == [] or rangex == []:
            M3[k] = 0
        else:
            M3[k] = np.sum(A[min(rangey):max(rangey)+1, min(rangex):max(rangex)+1] * A[min(rangey1):max(rangey1)+1, min(rangex1):max(rangex1)+1] * A[min(rangey2):max(rangey2)+1, min(rangex2):max(rangex2)+1])/np.size(A)

    return M3

#    sA = np.shape(A);
#    n3 = np.shape(list3)[0];
#    M3 = np.zeros(n3);

#    for k in range(n3):
#        shift1 = list3[k][0];
#        shift2 = list3[k][1];
#        A_padded_1 = np.pad(A,((shift1[0], 0), (shift1[1], 0)), mode='constant');
#        A_padded_1 = A_padded_1[0:sA[0], 0:sA[1]];
#        A_padded_2 = np.pad(A,((shift2[0], 0), (shift2[1], 0)), mode='constant');
#        A_padded_2 = A_padded_2[0:sA[0], 0:sA[1]];
#        M3[k] = np.sum(np.multiply(np.multiply(A, A_padded_1), A_padded_2))/np.size(A);
#    return M3;

def M2_2d(A, shift1):
    """Calculate second order 2-d autocorrelation of A for shift1.

    Keyword arguments:
    A -- the 2-d signal
    shift1 -- a tuple containing the shift
    """
    dim1, dim2, _ = np.shape(A)
    
    shift1y = -shift1[0]
    shift1x = -shift1[1]

    valsy1 = [0, -shift1y]
    valsx1 = [0, -shift1x]

    rangey = list(range(max(valsy1), dim1+min(valsy1)))
    rangey1 = [x + shift1y for x in rangey]
    rangex = list(range(max(valsx1), dim2+min(valsx1)))
    rangex1 = [x + shift1x for x in rangex]

    return np.sum(A[min(rangey):max(rangey)+1, min(rangex):max(rangex)+1, :] * A[min(rangey1):max(rangey1)+1, min(rangex1):max(rangex1)+1, :], axis=(0,1))/(dim1*dim2)

def M3_2d(A, shift1, shift2):
    """Calculate second order 3-d autocorrelation of A for shift1, shift2.

    Keyword arguments:
    A -- the 2-d signal
    shift1, shift2 -- tuples containing the shifts
    """
    dim1, dim2, _ = np.shape(A)
    
    shift1y = -shift1[0]
    shift1x = -shift1[1]

    shift2y = -shift2[0]
    shift2x = -shift2[1]

    valsy1 = [0, -shift1y, -shift2y]
    valsx1 = [0, -shift1x, -shift2x]

    rangey = list(range(max(valsy1), dim1+min(valsy1)))
    if rangey == []: return 0
    rangey1 = [x + shift1y for x in rangey]
    rangey2 = [x + shift2y for x in rangey]

    rangex = list(range(max(valsx1), dim2+min(valsx1)))
    if rangex == []: return 0
    rangex1 = [x + shift1x for x in rangex]
    rangex2 = [x + shift2x for x in rangex]
    
    return np.sum(A[min(rangey):max(rangey)+1, min(rangex):max(rangex)+1, :] * A[min(rangey1):max(rangey1)+1, min(rangex1):max(rangex1)+1, :] * A[min(rangey2):max(rangey2)+1, min(rangex2):max(rangex2)+1, :], axis=(0,1))/(dim1*dim2)

def M2_2d_grad(X, shift1):
    dim1, dim2, K = np.shape(X)
    
    shift1y = -shift1[0]
    shift1x = -shift1[1]

    valsy1 = [0, -shift1y]
    valsx1 = [0, -shift1x]

    rangey = list(range(max(valsy1), dim1+min(valsy1)))
    if rangey == []: return np.zeros((dim1, dim2, K))
    rangey1 = [x + shift1y for x in rangey]
    
    rangex = list(range(max(valsx1), dim2+min(valsx1)))
    if rangex == []: return np.zeros((dim1, dim2, K))
    rangex1 = [x + shift1x for x in rangex]

    X1 = X[min(rangey):max(rangey)+1, min(rangex):max(rangex)+1, :]
    X2 = X[min(rangey1):max(rangey1)+1, min(rangex1):max(rangex1)+1, :]

    T1 = np.zeros((dim1, dim2, K))
    T1[min(rangey):max(rangey)+1, min(rangex):max(rangex)+1, :] = X2
    T2 = np.zeros((dim1, dim2, K))
    T2[min(rangey1):max(rangey1)+1, min(rangex1):max(rangex1)+1, :] = X1
    return (T1 + T2)/(dim1*dim2)

def M3_2d_grad(X, shift1, shift2):
    dim1, dim2, K = np.shape(X)
    
    shift1y = -shift1[0]
    shift1x = -shift1[1]

    shift2y = -shift2[0]
    shift2x = -shift2[1]

    valsy1 = [0, -shift1y, -shift2y]
    valsx1 = [0, -shift1x, -shift2x]

    rangey = list(range(max(valsy1), dim1+min(valsy1)))
    if rangey == []: return np.zeros((dim1, dim2, K))
    rangey1 = [x + shift1y for x in rangey]
    rangey2 = [x + shift2y for x in rangey]

    rangex = list(range(max(valsx1),dim2+min(valsx1)))
    if rangex == []: return np.zeros((dim1, dim2, K))
    rangex1 = [x + shift1x for x in rangex]
    rangex2 = [x + shift2x for x in rangex]

    X1 = X[min(rangey):max(rangey)+1, min(rangex):max(rangex)+1]
    X2 = X[min(rangey1):max(rangey1)+1, min(rangex1):max(rangex1)+1]
    X3 = X[min(rangey2):max(rangey2)+1, min(rangex2):max(rangex2)+1]
    X1X2 = X1 * X2
        
    T1 = np.zeros((dim1, dim2, K))
    T1[min(rangey):max(rangey)+1, min(rangex):max(rangex)+1] = X2 * X3
    T2 = np.zeros((dim1, dim2, K))
    T2[min(rangey1):max(rangey1)+1, min(rangex1):max(rangex1)+1] = X1 * X3
    T3 = np.zeros((dim1, dim2, K))
    T3[min(rangey2):max(rangey2)+1, min(rangex2):max(rangex2)+1] = X1X2
    return (T1 + T2 + T3)/(dim1*dim2)

def M2_2d_ac_grad(X, shift1):
    dim1, dim2, K = np.shape(X)
    
    shift1y = -shift1[0]
    shift1x = -shift1[1]

    valsy1 = [0, -shift1y]
    valsx1 = [0, -shift1x]

    rangey = list(range(max(valsy1), dim1+min(valsy1)))
    rangey1 = [x + shift1y for x in rangey]
    rangex = list(range(max(valsx1), dim2+min(valsx1)))
    rangex1 = [x + shift1x for x in rangex]
    
    X1 = X[min(rangey):max(rangey)+1, min(rangex):max(rangex)+1, :]
    X2 = X[min(rangey1):max(rangey1)+1, min(rangex1):max(rangex1)+1, :]

    T1 = np.zeros((dim1, dim2, K))
    T1[min(rangey):max(rangey)+1, min(rangex):max(rangex)+1, :] = X2
    T2 = np.zeros((dim1, dim2, K))
    T2[min(rangey1):max(rangey1)+1, min(rangex1):max(rangex1)+1, :] = X1
    
    return (np.sum(X1 * X2, axis=(0,1))/(dim1*dim2), (T1 + T2)/(dim1*dim2))

def M3_2d_ac_grad(X, shift1, shift2):
    dim1, dim2, K = np.shape(X)
    
    shift1y = -shift1[0]
    shift1x = -shift1[1]

    shift2y = -shift2[0]
    shift2x = -shift2[1]

    valsy1 = [0, -shift1y, -shift2y]
    valsx1 = [0, -shift1x, -shift2x]

    rangey = list(range(max(valsy1), dim1+min(valsy1)))
    if rangey == []: return (0, np.zeros((dim1, dim2, K)))
    rangey1 = [x + shift1y for x in rangey]
    rangey2 = [x + shift2y for x in rangey]

    rangex = list(range(max(valsx1), dim2+min(valsx1)))
    if rangex == []: return (0, np.zeros((dim1, dim2, K)))
    rangex1 = [x + shift1x for x in rangex]
    rangex2 = [x + shift2x for x in rangex]
    
    X1 = X[min(rangey):max(rangey)+1, min(rangex):max(rangex)+1]
    X2 = X[min(rangey1):max(rangey1)+1, min(rangex1):max(rangex1)+1]
    X3 = X[min(rangey2):max(rangey2)+1, min(rangex2):max(rangex2)+1]
    X1X2 = X1 * X2
        
    T1 = np.zeros((dim1, dim2, K))
    T1[min(rangey):max(rangey)+1, min(rangex):max(rangex)+1] = X2 * X3
    T2 = np.zeros((dim1, dim2, K))
    T2[min(rangey1):max(rangey1)+1, min(rangex1):max(rangex1)+1] = X1 * X3
    T3 = np.zeros((dim1, dim2, K))
    T3[min(rangey2):max(rangey2)+1, min(rangex2):max(rangex2)+1] = X1X2
    
    return (np.sum(X1X2*X3, axis=(0,1))/(dim1*dim2), (T1 + T2 + T3)/(dim1*dim2))
