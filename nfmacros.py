import numpy as np
import sys


def stop():
    sys.exit()
    return

def printi():
    fsock = open('out.log','w')
    fsock.close()
    return

def printm(msg,x):

    x = np.array(x)
    s = str(x.shape)
    x = np.asmatrix(x)


    if np.isreal(x).all():
        printm1(msg,x,s)

    if np.iscomplex(x).any():
        printm1(msg + ' real',np.real(x),s)
        printm1(msg + ' imag',np.imag(x),s)

    return


  

def printm1(msg,x,s):

    m = x.shape[0]
    n = x.shape[1]

    printm0(msg,x,m,n,s)

    saveout = sys.stdout               
    fsock = open('out.log', 'a')
    sys.stdout = fsock                                      
    printm0(msg,x,m,n,s)
    sys.stdout = saveout                                   
    fsock.close()   

    return



def printm0(msg,x,m,n,s):
    
    sys.stdout.write("  ")
    sys.stdout.write(msg) 
    sys.stdout.write(" ")
    sys.stdout.write(s) 
    sys.stdout.write(" =")
    for i in range(m):
        sys.stdout.write('\n[')
        for j in range(n):
            if (j>0) and (j%6 == 0):
                sys.stdout.write('\n ')
            sys.stdout.write('{:12.5E} '.format(x[i,j]))
        sys.stdout.write(']')
    sys.stdout.write('\n')
    sys.stdout.write('\n')
    return




##############################################################################
# Testing Code 

if __name__ == "__main__":

    np.random.seed(20190906)
    n = 10
    a = np.random.randn(n,n)

    printm('a',a)
