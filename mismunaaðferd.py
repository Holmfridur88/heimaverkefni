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
from mpl_toolkits import mplot3d

u = lambda x: 0.5 * (2.0 + np.exp(2.0 - x) - np.exp(x))

def gildi(lamb):
    p = lambda x,y: 0.0 * x + 0.0 * y + 1.0
    q = lambda x,y: 0.0 * x + 0.0 * y - lamb ** 2
    f = lambda x,y: 0.0 * x + 0.0 * y

    a0 = q
    a1 = lambda x,y: 0.0 * x + 0.0 * y
    a2 = lambda x,y: 0.0 * x + 0.0 * y - 1.0
    
    return [p, q, f, a0, a1, a2]

# Program 8.5 Finite difference solver for 2D Poisson equation
# with Dirichlet boundary conditions on a rectangle
# Input: rectangle domain [xl,xr]x[yb,yt] with MxN space steps
# Output: matrix w holding solution values
def helmholtzeq(h, xl, xr, yb, yt, g1, g2, g3, g4, f):
    N = int((xr-xl)/h); M = int((yt-yb)/h); 
    m = M+1; n = N+1; mn = m*n
    x = np.linspace(xl, xr, n)
    y = np.linspace(yb, yt, m)
    A = np.zeros((mn,mn))
    b = np.zeros(mn)
    for i in range(1,m-1):  # interior points
        for j in range(1,m-1):
            A[i+j*m, i-1+j*m] = 1/h**2
            A[i+j*m, i+1+j*m] = 1/h**2
            A[i+j*m, i+j*m] = -2/h**2-2/h**2
            A[i+j*m, i+(j-1)*m] = 1/h**2
            A[i+j*m, i+(j+1)*m] = 1/h**2
            b[i+j*m] = f(x[i], y[j])
    for i in range(m):      # bottom and top boundary points
        j = 0
        A[i+j*m, i+j*m] = 1
        b[i+j*m] = g1(x[i])
        j = n-1
        A[i+j*m, i+j*m] = 1
        b[i+j*m] = g2(x[i])
    for j in range(1,n-1):  # left and right boundary points
        i = 0
        A[i+j*m, i+j*m] = 1
        b[i+j*m] = g3(y[j])
        i = m-1
        A[i+j*m, i+j*m] = 1
        b[i+j*m] = g4(y[j])
    c = spsolve(A,b)        # solve for solution in u labeling
    w = c.reshape(m, n)     # translate from u to w
#    plt.contourf(x,y,w)
    X,Y = np.meshgrid(x,y)
    ax = plt.axes(projection='3d')
    ax.plot_surface(X, Y, w, rstride=1, cstride=1,
                    cmap='viridis', edgecolor='none')
    ax.set_title('surface');
    return w

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

xl = 0; xr = 1; yb = 1; yt = 2 
h = 0.1
g1 = lambda x: np.log(x**2+1)   # define boundary values on bottom
g2 = lambda x: np.log(x**2+4)   # top
g3 = lambda y: 2*np.log(y)      # left side
g4 = lambda y: np.log(y**2+1)   # right side
f  = lambda x,y: 0.0*x+0.0*y    # define input function data

helmholtzeq(h, xl, xr, yb, yt, g1, g2, g3, g4, f)


