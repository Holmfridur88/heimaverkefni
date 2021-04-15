# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 13:15:24 2021

@author: Notandi
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import spsolve
import scipy.sparse as sp
from mpl_toolkits import mplot3d

# Program 8.5 Finite difference solver for 2D Poisson equation
# with Dirichlet boundary conditions on a rectangle
# Input: rectangle domain [xl,xr]x[yb,yt] with MxN space steps
# Output: matrix w holding solution values
def helmholtzeq(h, L1, L2, v, w, alpha, beta, g, q, f):
    N = int(L1/h); M = int(L2/h); 
    n = N+1; m = M+1; mn = m*n
    x = np.linspace(0, L1, n)
    y = np.linspace(0, L2, m)
#    A = np.zeros((mn,mn))
    A = sp.csr_matrix((mn, mn), dtype=np.int8).toarray()
    b = np.zeros(mn)
    for i in range(1,n-1):  # interior points
        for j in range(1,m-1):
            A[i+j*n, i+j*n-1] = -1/h**2
            A[i+j*n, i+j*n+1] = -1/h**2
            A[i+j*n, i+j*n] = 4/h**2-q(x[i], y[j])
            A[i+j*n, i+(j-1)*n] = -1/h**2
            A[i+j*n, i+(j+1)*n] = -1/h**2
            b[i+j*n] = f(x[i], y[j])
    for i in range(n):      # bottom and top boundary points
        j = 0       # bottom
        A[i+j*n, i+j*n] = 1
        b[i+j*n] = w(x[i])
        j = m-1     # top
        A[i+j*n, i+j*n] = 1
        b[i+j*n] = v(x[i])
    for j in range(1,m-1):  # left and right boundary points
        i = 0       # left
        A[i+j*n, i+j*n] = 4/h**2+q(x[i], y[j])+2/h*alpha(x[i], y[j])/beta(x[i], y[j])
        A[i+j*n, i+j*n+1] = -2/h**2
        A[i+j*n, i+(j-1)*n] = -1/h**2
        A[i+j*n, i+(j+1)*n] = -1/h**2
        b[i+j*n] = f(x[i], y[j])+2/h**2*g(y[j])/beta(x[i], y[j])
        i = n-1     # right
        A[i+j*n, i+j*n] = 4/h**2+q(x[i], y[j])+2/h*alpha(x[i], y[j])/beta(x[i], y[j])
        A[i+j*n, i+j*n-1] = -2/h**2
        A[i+j*n, i+(j-1)*n] = -1/h**2
        A[i+j*n, i+(j+1)*n] = -1/h**2
        b[i+j*n] = f(x[i], y[j])+2/h**2*g(y[j])/beta(x[i], y[j])
    c = spsolve(A,b)        # solve for solution in c labeling
    HZ = c.reshape(m, n)     # translate from c to HZ
    print(HZ)
    X,Y = np.meshgrid(x,y)
    ax = plt.axes(projection='3d')
    ax.plot_surface(X, Y, HZ, rstride=1, cstride=1,
                    cmap='viridis', edgecolor='none')
    ax.set_title('surface');
    ax.set_xlabel("X axis label")
    ax.set_ylabel("Y axis label")
    ax.set_zlabel("Z axis label")
    
    return HZ

# =============================================================================

lamb = 1
p = lambda x,y: 0.0 * x + 0.0 * y + 1.0
q = lambda x,y: 0.0 * x + 0.0 * y - lamb ** 2
f = lambda x,y: 0.0 * x + 0.0 * y
w = lambda x:   0.0 * x + 1.0
v = lambda x:   0.0 * x
a = lambda x,y: 0.0 * x + 0.0 * y               # alpha
b = lambda x,y: 0.0 * x + 0.0 * y + 1.0         # beta
g = lambda y:   0.0 * y                         # gamma
L1 = 1; L2 = 1
h = 1/4

HZ = helmholtzeq(h, L1, L2, v, w, a, b, g, q, f)


