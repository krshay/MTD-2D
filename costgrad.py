# from mpi4py import MPI
import numpy as np
import time
import sys
from nfmacros import printi, printm, stop
import scipy.special as spl
import scipy.sparse as sp
import matplotlib.pyplot as plt
####### To avoid issues with matplotlib wayland on Fedora 29
######import os
######os.putenv("QT_QPA_PLATFORM","xcb")
######os.putenv("XDG_SESSION_TYPE","x11")

##########################################################################
#
#   This file has four main functions:
#
#   f, f0 = make_test_func(n)
#
#   Mat = precomp(kvals,roots,nu,n1,T)
#
#   s1h, s2h, s3h = comp_moments(c,Mat,comm)
#
#   G1, G2, G3 = comp_grad(c,Mat,comm)
#
#
##########################################################################

# def comp_grad(c,Mat,comm):

#     nproc = comm.Get_size()
#     rank = comm.Get_rank()

#     T = Mat['T']
#     Ta = np.conj(sp.csr_matrix.transpose(T))
#     z = sp.csr_matrix.dot(Ta,c)

#     kvals = Mat['kvals']
#     roots = Mat['roots']
#     nu = z.shape[0]
#     n = Mat['n']
#     Psi = Mat['Psi']
#     PsiT = Mat['PsiT']

#     Eta1 = Mat['Eta1']

#     Zeta1 = Mat['Zeta1']
#     Zeta2 = Mat['Zeta2']

#     Zeta1T = Mat['Zeta1T']
#     Zeta2T = Mat['Zeta2T']

#     Phi1 = Mat['Phi1']
#     Phi2 = Mat['Phi2']
#     Phi3 = Mat['Phi3'] 

#     Phi1T = Mat['Phi1T']
#     Phi2T = Mat['Phi2T']
#     Phi3T = Mat['Phi3T'] 

#     # First order
#     nq = Zeta1.shape[0]
#     nq1 = nq//2
#     g = np.zeros((nq,1))
#     g[nq1] = 1
#     g = sp.csr_matrix.dot(Zeta1T,g)
#     G1 = sp.csr_matrix.dot(PsiT[0,0],g)
#     G1 = np.transpose(G1)

#     # Second order
#     C1 = np.multiply(Eta1,z)
#     C1 = sp.csr_matrix.dot(Zeta1,C1)
#     C1 = sp.csr_matrix.dot(Zeta2T,C1)
#     C1 = np.multiply(Eta1,C1)
#     C1 = np.transpose(C1)

#     C2 = np.multiply(Eta1,z)
#     C2 = sp.csr_matrix.dot(Zeta2,C2)
#     C2 = sp.csr_matrix.dot(Zeta1T,C2)
#     C2 = np.multiply(Eta1,C2)
#     C2 = np.transpose(C2)

#     G2 = np.reshape(C1 + C2,(n,n,nu))

#     # precomp for third order
#     nqq = Phi1.shape[0]

#     A1 = np.multiply(Eta1,z)
#     A1 = sp.csr_matrix.dot(Zeta1,A1)
#     A1 = sp.csr_matrix.dot(Phi1,A1)
#     A1 = np.reshape(A1,(-1,n,n))

#     A2 = np.multiply(Eta1,z)
#     A2 = sp.csr_matrix.dot(Zeta1,A2)
#     A2 = sp.csr_matrix.dot(Phi2,A2)
#     A2 = np.reshape(A2,(-1,n,n))

#     A3 = np.multiply(Eta1,z)
#     A3 = sp.csr_matrix.dot(Zeta1,A3)
#     A3 = sp.csr_matrix.dot(Phi3,A3)
#     A3 = np.reshape(A3,(-1,n,n))

#     C1 = sp.csr_matrix.dot(Zeta1,Eta1)
#     C1 = sp.csr_matrix.dot(Phi1,C1)
#     C1 = np.reshape(C1,(-1,n,n))

#     C2 = sp.csr_matrix.dot(Zeta1,Eta1)
#     C2 = sp.csr_matrix.dot(Phi2,C2)
#     C2 = np.reshape(C2,(-1,n,n))

#     C3 = sp.csr_matrix.dot(Zeta1,Eta1)
#     C3 = sp.csr_matrix.dot(Phi3,C3)
#     C3 = np.reshape(C3,(-1,n,n))

#     # Third order
#     t0 = MPI.Wtime()
#     block_sz = n**2 // nproc + 1
#     mini_G3 = np.zeros((block_sz,n,n,nu), dtype=complex)
#     ix = np.zeros((n,n)).astype(np.int)
#     jx = np.zeros((n,n)).astype(np.int)
#     for k in range(block_sz):
#         k0 = rank*block_sz + k
#         if (k0 > n**2 - 1):
#             break
#         i = k0 // n
#         j = k0 % n
#         for i1 in range(n):
#             for j1 in range(n):
#                 i2 = (-i - i1) % n
#                 j2 = (-j - j1) % n
#                 ix[i1,j1] = i2
#                 jx[i1,j1] = j2
        
#         D1 = A3[:,ix,jx]
#         D1 = np.multiply(A2,D1)
#         D1 = np.reshape(D1,(-1,n**2))
#         D1 = sp.csr_matrix.dot(Phi1T,D1)
#         D1 = sp.csr_matrix.dot(Zeta1T,D1)
#         D = np.reshape(Eta1,(-1,n,n))
#         D = np.reshape(D[:,i,j],(-1,1))
#         D1 = np.multiply(D1,D)
#         D1 = np.transpose(D1)
#         D1 = np.reshape(D1,(n,n,-1))

#         D2 = A3[:,ix,jx]
#         D2 = np.multiply(D2,np.reshape(A1[:,i,j],(-1,1,1)))
#         D2 = np.reshape(D2,(-1,n**2))
#         D2 = sp.csr_matrix.dot(Phi2T,D2)
#         D2 = sp.csr_matrix.dot(Zeta1T,D2)
#         D = np.reshape(Eta1,(-1,n**2))
#         D2 = np.multiply(D2,D)
#         D2 = np.transpose(D2)
#         D2 = np.reshape(D2,(n,n,-1))

#         D3 = np.multiply(A2,np.reshape(A1[:,i,j],(-1,1,1)))
#         D3 = np.reshape(D3,(-1,n**2))
#         D3 = sp.csr_matrix.dot(Phi3T,D3)
#         D3 = sp.csr_matrix.dot(Zeta1T,D3)
#         D = np.reshape(Eta1,(-1,n,n))
#         D = D[:,ix,jx]
#         D = np.reshape(D,(-1,n**2))
#         D3 = np.multiply(D3,D)
#         D3 = np.transpose(D3)
#         D3 = np.reshape(D3,(n,n,-1))
#         mini_G3[k,:,:,:] = D1 + D2 + D3

#     G1 = np.transpose(G1)  
#     G1 = sp.csr_matrix.dot(np.conj(T),G1)
#     G1 = np.real(np.transpose(G1))

#     G2 = np.reshape(G2,(n**2,nu))
#     G2 = np.transpose(G2)  
#     G2 = sp.csr_matrix.dot(np.conj(T),G2)
#     G2 = np.real(np.transpose(G2))
#     G2 = np.reshape(G2,(n,n,nu))

#     mini_G3 = np.reshape(mini_G3,(block_sz*n**2,nu))
#     mini_G3 = np.transpose(mini_G3)
#     mini_G3 = sp.csr_matrix.dot(np.conj(T),mini_G3)
#     mini_G3 = np.real(np.transpose(mini_G3))
#     mini_G3 = np.reshape(mini_G3,(block_sz,n,n,nu))

#     commG3 = np.zeros(n**4 * nu)
#     G3 = np.zeros(n**4 * nu)
#     for k in range(block_sz):
#         k0 = rank*block_sz + k
#         if (k0 > n**2 - 1):
#             break
#         commG3[k0*n**2*nu:(k0+1)*n**2*nu] = mini_G3[k,:,:,:].flatten()
 
#     comm.Allreduce(commG3, G3, op=MPI.SUM)
#     G3 = np.reshape(G3,(n,n,n,n,nu))
    
#     return G1, G2, G3

# #
# #
# #######################################################################
# #
# #

# def precomp(kvals,roots,nu,n1,T):

#     # radius of the circular support
#     n2 = n1 // 2

#     # edge length of the zero-padded image
#     n = 4*n2 + 1

#     r = np.zeros((n1,n1))
#     t = np.zeros((n1,n1))
#     idx = np.zeros((n1,n1)).astype(np.int)
#     jdx = np.zeros((n1,n1)).astype(np.int)
#     for i in range(n1):
#         for j in range(n1):
#             y = (i - n2)/n2
#             x = (j - n2)/n2
#             r[i,j] = np.sqrt(x**2 + y**2)
#             t[i,j] = np.arctan2(y,x)
#             if r[i,j] >= 1:
#                 idx[i,j] = i
#                 jdx[i,j] = j

#     B = np.zeros((n1,n1,nu)).astype(np.complex)
#     for ii in range(nu):
#         B[:,:,ii] = np.multiply(spl.jv(np.abs(kvals[ii]),roots[ii]*r), \
#             np.exp(1j*kvals[ii]*t))
#         B[idx,jdx,ii] = 0

#     B = np.reshape(B,(n1**2,nu))
#     r = np.reshape(r,(n1**2,1))
#     t = np.reshape(t,(n1**2,1))
#     Bp = full2supp(B,n1)
#     rp = full2supp(r,n1)
#     tp = full2supp(t,n1)
#     N = tp.shape[0]
   
#     # Initialize 
#     s1 = 0 
#     s2 = np.zeros((n,n))
#     fa = np.zeros((n1,n1,nu))
#     Bf = np.reshape(B,(n1,n1,nu))

#     # Pre-computation
#     af = np.zeros((n,n,nu)).astype(np.complex)
#     for ii in range(nu):
#         ff = np.zeros((n,n)).astype(np.complex)
#         ff[n2:n2+n1,n2:n2+n1] = Bf[:,:,ii]
#         af[:,:,ii] = np.fft.fftn(np.fft.ifftshift(ff))

#     kuniq = np.sort(np.unique(kvals))
#     nq = len(kuniq)
#     kmap = {}
#     for ii in range(nq):
#         kmap[kuniq[ii]] = ii

#     v = np.zeros(nu)    
#     iv1 = np.zeros(nu)
#     iv2 = np.zeros(nu)
#     jv = np.zeros(nu)
#     jv = np.zeros(nu)
#     for ii in range(nu):
#         v[ii] = 1
#         iv1[ii] = kmap[kvals[ii,0]]
#         iv2[ii] = kmap[-kvals[ii,0]]
#         jv[ii] = ii
#     Zeta1 = sp.csr_matrix((v,(iv1,jv)),shape=(nq, nu))
#     Zeta2 = sp.csr_matrix((v,(iv2,jv)),shape=(nq, nu))

#     Psi = {}
#     PsiT = {}
#     for i in range(n):
#         for j in range(n):
#             v = np.zeros(nu).astype(np.complex)
#             iv = np.zeros(nu).astype(np.int)
#             jv = np.zeros(nu).astype(np.int)
#             for ii in range(nu):
#                 v[ii] = af[i,j,ii]
#                 iv[ii] = ii
#                 jv[ii] = ii
#             P = sp.csr_matrix((v,(iv,jv)),shape=(nu, nu))
#             Psi[i,j] = P
#             PsiT[i,j] = sp.csr_matrix.transpose(P)

#     Eta1 = np.zeros((nu,n**2)).astype(np.complex)
#     jj = 0
#     for i in range(n):
#         for j in range(n):
#             for ii in range(nu):
#                 Eta1[ii,jj] = af[i,j,ii]
#             jj += 1

#     ik1 = np.zeros((nq**2,1)).astype(np.int)
#     ik2 = np.zeros((nq**2,1)).astype(np.int)
#     ik3 = np.zeros((nq**2,1)).astype(np.int)

#     ii = 0
#     nq1 = nq//2
#     for k1 in range(nq):
#         for k2 in range(nq):
#             k3 = nq1 - (k1-nq1 + k2-nq1)
#             if (k3 < 0) or (k3 >= nq):
#                 continue
#             ik1[ii] = k1
#             ik2[ii] = k2
#             ik3[ii] = k3
#             ii += 1
#     ik1 = ik1[0:ii]
#     ik2 = ik2[0:ii]
#     ik3 = ik3[0:ii]
#     nw = ii

#     v = np.zeros(nw)
#     iv = np.zeros(nw).astype(np.int)
#     jv1 = np.zeros(nw).astype(np.int)
#     jv2 = np.zeros(nw).astype(np.int)
#     jv3 = np.zeros(nw).astype(np.int)
#     for ii in range(nw): 
#         v[ii] = 1
#         iv[ii] = ii
#         jv1[ii] = ik1[ii]
#         jv2[ii] = ik2[ii]
#         jv3[ii] = ik3[ii]

#     Phi1 = sp.csr_matrix((v,(iv,jv1)),shape=(nw, nq))
#     Phi2 = sp.csr_matrix((v,(iv,jv2)),shape=(nw, nq))
#     Phi3 = sp.csr_matrix((v,(iv,jv3)),shape=(nw, nq))

#     Mat = {}

#     Mat['T'] = T
    
#     Mat['Psi'] = Psi
#     Mat['PsiT'] = PsiT

#     Mat['Eta1'] = Eta1

#     Mat['Zeta1'] = Zeta1
#     Mat['Zeta2'] = Zeta2

#     Mat['Zeta1T'] = sp.csr_matrix.transpose(Zeta1)
#     Mat['Zeta2T'] = sp.csr_matrix.transpose(Zeta2)

#     Mat['Phi1'] = Phi1
#     Mat['Phi2'] = Phi2
#     Mat['Phi3'] = Phi3
#     Mat['Phi1T'] = sp.csr_matrix.transpose(Phi1)
#     Mat['Phi2T'] = sp.csr_matrix.transpose(Phi2)
#     Mat['Phi3T'] = sp.csr_matrix.transpose(Phi3)

#     Mat['kvals'] = kvals
#     Mat['roots'] = roots

#     Mat['n'] = n
   
#     return Mat

# #
# #
# #############################################################################
# #
# #

# def comp_moments(c,Mat,comm):

#     nproc = comm.Get_size()
#     rank = comm.Get_rank()

#     T = Mat['T']
#     Ta = np.conj(sp.csr_matrix.transpose(T)) 
#     z = sp.csr_matrix.dot(Ta,c)
    
#     n = Mat['n']
#     kvals = Mat['kvals']
#     roots = Mat['roots']
#     nu = z.shape[0]

#     Psi = Mat['Psi']
#     Zeta1 = Mat['Zeta1']
#     Zeta2 = Mat['Zeta2']

#     Phi1 = Mat['Phi1']
#     Phi2 = Mat['Phi2']
#     Phi3 = Mat['Phi3'] 

#     Eta1 = Mat['Eta1']

#     ## First moment
#     nq = Zeta1.shape[0]
#     nq1 = nq//2
#     g2 = np.zeros((nq,1))
#     g2[nq1] = 1
#     g1 = sp.csr_matrix.dot(Psi[0,0],z)
#     g1 = sp.csr_matrix.dot(Zeta1,g1)
#     s1h = np.sum(np.multiply(g1,g2))
#     s1h = np.real(s1h)

#     # Second moment
#     C = np.multiply(Eta1,z)
#     C1 = sp.csr_matrix.dot(Zeta1,C)
#     C2 = sp.csr_matrix.dot(Zeta2,C)
#     C3 = np.multiply(C1,C2)
#     C3 = np.sum(C3,axis=0)
#     s2h = np.reshape(C3,(n,n))
#     s2h = np.real(s2h)

#     # precomp for third moment
#     s3 = np.zeros((n,n,n,n))

#     A = np.multiply(Eta1,z)
#     A = sp.csr_matrix.dot(Zeta1,A)
#     A1 = sp.csr_matrix.dot(Phi1,A)
#     A1 = np.reshape(A1,(-1,n,n))

#     A = np.multiply(Eta1,z)
#     A = sp.csr_matrix.dot(Zeta1,A)
#     A2 = sp.csr_matrix.dot(Phi2,A)
#     A2 = np.reshape(A2,(-1,n,n))

#     A = np.multiply(Eta1,z)
#     A = sp.csr_matrix.dot(Zeta1,A)
#     A3 = sp.csr_matrix.dot(Phi3,A)
#     A3 = np.reshape(A3,(-1,n,n))

#     # Third moment
#     block_sz = n**2 // nproc + 1
#     B = np.zeros(A3.shape).astype(np.complex)
#     mini_s3h = np.zeros((n, n), dtype=complex)
#     comm_s3h = np.zeros(n**4)
#     s3h = np.zeros(n**4)

#     ix = np.zeros((n,n)).astype(np.int)
#     jx = np.zeros((n,n)).astype(np.int)
#     for k in range(block_sz):
#         k0 = rank*block_sz + k
#         if (k0 > n**2 - 1):
#             break
#         i = k0 // n
#         j = k0 % n
#         for i1 in range(n):
#             for j1 in range(n):
#                 i2 = (-i - i1) % n
#                 j2 = (-j - j1) % n
#                 ix[i1,j1] = i2
#                 jx[i1,j1] = j2
#         B = A3[:,ix,jx]
#         B = np.multiply(A2,B)
#         B = np.multiply(B,np.reshape(A1[:,i,j],(-1,1,1)))
#         mini_s3h = np.sum(B,axis=0)
#         comm_s3h[k0*n**2:(k0+1)*n**2] = np.real(mini_s3h.flatten())

#     comm.Allreduce(comm_s3h, s3h, op=MPI.SUM)
#     s3h = np.reshape(s3h, (n,n,n,n))

#     return s1h, s2h, s3h

#
#
###########################################################################
#
#

def check_moments(B,z,m,n,nu,kvals):

    n1 = n//2 + 1
    n2 = n//4

    s1 = 0
    s2 = np.zeros((n,n))
    s3 = np.zeros((n,n,n,n))
    Br = np.zeros((n1**2,nu)).astype(np.complex)
    for k in range(m):
        t = 2*np.pi*k/m
        # for jj in range(n1**2):
        #     for ii in range(nu):
        #         Br[jj,ii] = B[jj,ii]*np.exp(1j*kvals[ii]*t)
        Br = B*np.exp(1j*kvals*t)
        
        fr0 = np.real(np.dot(Br,z))
        fr0 = np.reshape(fr0,(n1,n1))
        fr = np.zeros((n,n))
        fr[n2:n2+n1,n2:n2+n1] = fr0

        #plt.figure()
        #plt.imshow(fr)
        #plt.colorbar()

        a = np.fft.fftn(np.fft.ifftshift(fr))
        c1 = np.real(a[0,0])
        s1 += c1

        c2h = np.abs(a)**2
        c2 = np.real(np.fft.ifftn(c2h))
        s2 += c2

        c3h = np.zeros((n,n,n,n)).astype(np.complex)
        for i in range(n):
            for j in range(n):
                for i1 in range(n):
                    for j1 in range(n):
                        i2 = (-i - i1) % n
                        j2 = (-j - j1) % n
                        c3h[i,j,i1,j1] = a[i,j]*a[i1,j1]*a[i2,j2]
        c3 = np.real(np.fft.ifftn(c3h))
        s3 += c3#[ :n1, :n1, :n1, :n1]

    s1 /= m
    s2 /= m
    s3 /= m

    return s1, s2, s3

#
#
##########################################################################
#
#

def full2supp(A,n1):

    n2 = n1//2
    mask = np.zeros((n1,n1)).astype(bool)
    for i in range(n1):
        for j in range(n1):
            y1 = (i - n2)/n2
            x1 = (j - n2)/n2
            r = np.sqrt(x1**2 + y1**2)
            if r < 1:
                mask[i,j] = True
            else:
                mask[i,j] = False
    mask = mask.flatten()
    Ap = A[mask,:]
    return Ap

#
#
#######################################################################
#
#

def supp2full(Ap,n1):

    n2 = n1//2
    idx = np.zeros(n1**2).astype(np.int)
    jj = 0
    kk = 0
    for i in range(n1):
        for j in range(n1):
            y1 = (i - n2)/n2
            x1 = (j - n2)/n2
            r = np.sqrt(x1**2 + y1**2)
            if r < 1:
                idx[jj] = kk
                jj += 1
            kk += 1
    idx = idx[0:jj]

    m = Ap.shape[1]
    A = np.zeros((n1**2,m)).astype(Ap.dtype)
    A[idx,:] = Ap

    return A

def make_test_func(n):

    n1 = n//2
    n2 = n//4

    # Test function 
    f0 = np.zeros((n1,n1))
    for i in range(n1):
        for j in range(n1):
            x = (j-n2)/n2
            y = (i-n2)/n2
            r = np.sqrt(x**2+y**2)
            x1 = (x + y)/np.sqrt(2)
            if r >= 1:
                continue
            f0[i,j] = np.exp(-1/(1-r**2))*np.exp(1)*np.sin(x1**2*np.pi/2)
    
    f = np.zeros((n,n))
    f[n2:n2+n1,n2:n2+n1] = f0
    
    return f,f0

#
#
########################################################################
#
#

# Testing Code 
def test():

    np.random.seed(20190906)

    # Parameters
    n = 2**4
    m = 2**4
    ne = 2**4


    # Constants
    n1 = n//2
    n2 = n//4


    f, f0 = make_test_func(n)


    # Visualize test function
    #plt.figure()
    #plt.imshow(f)
    #plt.title('f')
    #plt.colorbar()
    #printm('f0',f0)
    #printm('f',f) 

    c, roots, kvals, nu, T, B, z = comp_rep(f0,n1,ne)
    yz = np.real(np.dot(B,z))

    #printm('yz',yz)
    
    f0a = np.reshape(yz,(n1,n1))

    #printm('z',z)
    #printm('roots',roots)
    #printm('kvals',kvals)
    y = np.reshape(f0,(n1**2,1))
    #printm('y',y)
    #printm('yz',yz)
    errz = np.abs(y -yz)
    relz = np.linalg.norm(errz)/np.linalg.norm(y)
    #printm('errz',errz)
    #printm('relz',relz)


    # Visualize approximation
    #plt.figure()
    #plt.imshow(f0a)
    #plt.colorbar()
    #plt.title('f0a')
    #plt.figure()
    #plt.imshow(f0)
    #plt.colorbar()
    #plt.title('f0')
    

    # Compute moments using fft and averaging over m rotations as check
  
    s1check, s2check, s3check = check_moments(B,z,m,n,nu,kvals)

    # Compute moments in terms of coefficients

    t0 = time.time()
    Mat = precomp(kvals,roots,nu,n,T)
    t1 = time.time()
    dt = t1 - t0
    printm('precomp dt',dt)
    
    t0 = time.time()
    s1h, s2h, s3h = comp_moments(c,Mat)
    t1 = time.time()
    dt = t1 - t0

    s1 = np.fft.ifftn(s1h)
    s2 = np.fft.ifftn(s2h)
    s3 = np.fft.ifftn(s3h)

    err1 = np.abs(s1check - s1)
    rel1 = np.linalg.norm(err1)/np.linalg.norm(s1check)
    printm('||check- coeff|| rel1',rel1)

    err2 = np.abs(s2check - s2)
    rel2 = np.linalg.norm(err2)/np.linalg.norm(s2check)
    printm('||check - coeff|| rel2',rel2)

    err3 = np.abs(s3check-s3)
    rel3 = np.linalg.norm(err3)/np.linalg.norm(s3check)
    printm('||check - coeff|| rel3',rel3)
    #printm('moment dt',dt)


    t0 = time.time()
    G1, G2, G3 = comp_grad(c,Mat)
    t1 = time.time()
    dt = t1 -t0
    printm('time grad',dt)

    tol = 10e-7
    eps = np.random.randn(nu,1)
    eps = eps*tol
    ce = c + eps
    s1h, s2h, s3h = comp_moments(c,Mat)
    s1he, s2he, s3he = comp_moments(ce,Mat)


    s1hg = s1h + np.dot(G1,eps)
    err1 = np.abs(s1hg - s1he)
    printm('tol',tol)
    printm('real err1/tol',err1/tol)

    v = np.dot(G2,eps)
    v = np.reshape(v,(n,n))
    s2hg = s2h + v
    err2 = np.abs(s2hg - s2he)
    rel2 = np.linalg.norm(err2)/np.linalg.norm(s2he)
    printm('real rel2/tol',rel2/tol)
  
    v = np.dot(G3,eps)
    v = np.reshape(v,(n,n,n,n))
    s3hg = s3h + v
    err3 = np.abs(s3hg - s3he)
    rel3 = np.linalg.norm(err3)/np.linalg.norm(s3he)
    printm('real rel3/tol',rel3/tol)


if __name__ == "__main__":
    test()
