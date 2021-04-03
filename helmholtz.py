# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 13:15:24 2021

@author: Notandi
"""

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
# from bvp_ode import *
import matplotlib.pyplot as plt

u = lambda x: 0.5 * (2.0 + np.exp(2.0 - x) - np.exp(x))

def gildi(lamb):
    p = lambda x,y: 0.0 * x + 0.0 * y + 1.0
    q = lambda x,y: 0.0 * x + 0.0 * y - lamb ** 2
    f = lambda x,y: 0.0 * x + 0.0 * y

    a0 = q
    a1 = lambda x,y: 0.0 * x + 0.0 * y
    a2 = lambda x,y: 0.0 * x + 0.0 * y - 1.0
    
    return [p, q, f, a0, a1, a2]

def fdm(h, ax, bx, ay, by, g1, g2, g3, g4, f):
    N = (bx-ax)/h; M = (by-ay)/h;    
    m = int(M+1); n = int(N+1); mn = m*n
    x = np.linspace(ax+h, bx-h, N)
    y = np.linspace(ay+h, by-h, M)
    A = np.zeros((mn,mn))
    b = np.zeros(mn)
    for i in range(1,m-1):
        for j in range(1,m-1):
            A[i+(j-1)*m, i-1+(j-1)*m] = 1/h**2
            A[i+(j-1)*m, i+1+(j-1)*m] = 1/h**2
            A[i+(j-1)*m, i+(j-1)*m] = -2/h**2-2/h**2
            A[i+(j-1)*m, i+(j-2)*m] = 1/h**2
            A[i+(j-1)*m, i+j*m] = 1/h**2
            b[i+(j-1)*m] = f(x[i], y[j])
    for i in range(m-1):          # bottom and top boundary points
        j = 1
        A[i+(j-1)*m, i+(j-1)*m] = 1
        b[i+(j-1)*m] = g1(x[i])
        j = n
        A[i+(j-1)*m, i+(j-1)*m] = 1
        b[i+(j-1)*m] = g2(x[i])
    for j in range(1,n-1):  	# left and right boundary points
        i=1
        A[i+(j-1)*m, i+(j-1)*m] = 1
        b[i+(j-1)*m] = g3(y[j])
        i = m
        A[i+(j-1)*m, i+(j-1)*m] = 1
        b[i+(j-1)*m] = g4(y[j])
    
    print(A)
    print(b)
    c = spsolve(A,b)    # solve for solution in u labeling
    w = c.reshape(m, n);     # translate from u to w
    plt.contourf(x,y,w)
    return c

# =============================================================================

#lamb = 1/100
#[p, q, f, a0, a1, a2] = gildi(lamb)

#alpha1 = 1.0
#beta1 = -1.0
#gamma1 = 0.0
#alpha2 = 1.0
#beta2 = 0.0
#gamma2 = 1.0

#N = 6

ax = 0; bx = 1; ay = 0; by = 1
h = 0.5
g1 = lambda x: np.sin(np.pi*x)
g2 = lambda x: np.sin(np.pi*x)
g3 = lambda y: 0.0*y
g4 = lambda y: 0.0*y
f  = lambda x,y: 0.0*x+0.0*y

fdm(h, ax, bx, ay, by, g1, g2, g3, g4, f)


