# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 13:15:24 2021

@author: Notandi
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import spsolve
from mpl_toolkits import mplot3d

# Program 8.5 Finite difference solver for 2D Poisson equation
# with Dirichlet boundary conditions on a rectangle
# Input: rectangle domain [xl,xr]x[yb,yt] with MxN space steps
# Output: matrix w holding solution values
def helmholtzeq(L1, L2, h, lamb, w, v):
    q = lambda x,y: 0*x + 0*y - lamb**2
    f = lambda x,y: 0*x + 0*y
    alpha = lambda x,y: 0*x + 0*y
    beta  = lambda x,y: 0*x + 0*y + 1
    gamma = lambda y:   0*y
    N = int(L1/h); M = int(L2/h); 
    n = N+1; m = M+1; mn = m*n
    x = np.linspace(0, L1, n)
    y = np.linspace(0, L2, m)
    A = np.zeros((mn,mn))
    b = np.zeros(mn)
    for i in range(1,n-1):  # interior points
        for j in range(1,m-1):
            A[i+j*n, i+j*n-1] = -1/h**2
            A[i+j*n, i+j*n+1] = -1/h**2
            A[i+j*n, i+j*n] = 4/h**2+q(x[i], y[j])
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
        b[i+j*n] = f(x[i], y[j])+2/h**2*gamma(y[j])/beta(x[i], y[j])
        i = n-1     # right
        A[i+j*n, i+j*n] = 4/h**2+q(x[i], y[j])+2/h*alpha(x[i], y[j])/beta(x[i], y[j])
        A[i+j*n, i+j*n-1] = -2/h**2
        A[i+j*n, i+(j-1)*n] = -1/h**2
        A[i+j*n, i+(j+1)*n] = -1/h**2
        b[i+j*n] = f(x[i], y[j])+2/h**2*gamma(y[j])/beta(x[i], y[j])
    c = np.linalg.solve(A,b)
    HZ = c.reshape(m,n)     # translate from c to HZ
    X,Y = np.meshgrid(x,y)
    return X, Y, HZ

def plot_surf(X, Y, Z):
    ax = plt.axes(projection='3d')
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='plasma')
    ax.set_title('surface')
    ax.set_xlabel("x axis")
    ax.set_ylabel("y axis")
    ax.set_zlabel("z axis")
    plt.show()
    
def plot_compare(X, Y, Z, u):
    U = u(X,Y)
    W = Z-U
    ax = plt.axes(projection='3d')
#    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='viridis')
#    ax.plot_surface(X, Y, U, rstride=1, cstride=1, cmap='viridis')
    ax.plot_surface(X, Y, W, rstride=1, cstride=1, cmap='viridis')
    ax.set_title('surface')
    ax.set_xlabel("x axis")
    ax.set_ylabel("y axis")
    ax.set_zlabel("z axis")

# =============================================================================

# UPPHAFSGILDI
lamb = [1/100, 1/10, 1, 10, 30]
L1 = 1; L2 = [1, 2]
h = [1/4, 1/20, 1/50]

# LIÐUR II
w1 = lambda x:   0*x + 1
v1 = lambda x:   0*x
X, Y, HZ = helmholtzeq(L1, L2[0], h[0], lamb[0], w1, v1)
print(HZ)


# LIÐUR III
for i in range(2,4):
    plt.figure(i-1)
    X, Y, Z = helmholtzeq(L1, L2[1], h[1], lamb[i], w1, v1)
    u = lambda x,y: 0*x + np.sin(lamb[i]*(L2[1]-y))/np.sin(lamb[i]*L2[1])
    plot_compare(X, Y, Z, u)


# LIÐUR IV
u0 = 10; u1 = 1
w2 = lambda x: -u0*x/L1*(x/L1-1)**2*(1+x/L1)
v2 = lambda x:  u1*x/L1*(1-x/L1)*(1+x/L1)**2

for i in range(len(lamb)):
    plt.figure(i+3)
    X, Y, Z = helmholtzeq(L1, L2[1], h[2], lamb[i], w2, v2)
    plot_surf(X, Y, Z)
