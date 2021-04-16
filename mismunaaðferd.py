# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 13:15:24 2021

@author: Notandi
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import spsolve
from mpl_toolkits import mplot3d

# Program 8.5 Finite difference solver for 2D PDE
# with von Neuman boundary conditions on a rectangle
# Input: rectangle domain [0,L1]x[0,L2] with NxM space steps
# Output: matrix HZ holding solution values
def helmholtzeq(L1, L2, h, lamb, w, v):
    q = lambda x,y: 0*x + 0*y - lamb**2
    N = int(L1/h); M = int(L2/h); 
    n = N+1; m = M+1; mn = m*n
    x = np.linspace(0, L1, n)
    y = np.linspace(0, L2, m)
    A = np.zeros((mn,mn))
    b = np.zeros(mn)
    for i in range(1,n-1):  # interior points
        for j in range(1,m-1):
            A[i+j*n, i+j*n]     =  4/h**2 + q(x[i], y[j])
            A[i+j*n, i+j*n-1]   = -1/h**2
            A[i+j*n, i+j*n+1]   = -1/h**2
            A[i+j*n, i+(j-1)*n] = -1/h**2
            A[i+j*n, i+(j+1)*n] = -1/h**2
            b[i+j*n] = 0
    for i in range(n):      # bottom and top boundary points
        j = 0       # bottom
        A[i+j*n, i+j*n]     = 1
        b[i+j*n] = w(x[i])
        j = m-1     # top
        A[i+j*n, i+j*n]     = 1
        b[i+j*n] = v(x[i])
    for j in range(1,m-1):  # left and right boundary points
        i = 0       # left
        A[i+j*n, i+j*n]     =  4/h**2 + q(x[i], y[j])
        A[i+j*n, i+j*n+1]   = -2/h**2
        A[i+j*n, i+(j-1)*n] = -1/h**2
        A[i+j*n, i+(j+1)*n] = -1/h**2
        b[i+j*n] = 0
        i = n-1     # right
        A[i+j*n, i+j*n]     =  4/h**2 + q(x[i], y[j])
        A[i+j*n, i+j*n-1]   = -2/h**2
        A[i+j*n, i+(j-1)*n] = -1/h**2
        A[i+j*n, i+(j+1)*n] = -1/h**2
        b[i+j*n] = 0
    c = np.linalg.solve(A,b)    # solve for solution in c labeling
    HZ = c.reshape(m,n)         # projecting c to a grid
    X,Y = np.meshgrid(x,y)
    return X, Y, HZ

def plot_surf(X, Y, Z):
    ax = plt.axes(projection='3d')
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='plasma')
    ax.set_title('Mynd af lausn verkefnisins')
    ax.set_xlabel("x axis")
    ax.set_ylabel("y axis")
    ax.set_zlabel("z axis")
    set_axes_equal(ax)
    plt.show()
    
def plot_compare(X, Y, Z, u):
    U = u(X,Y)
    W = Z-U
    ax = plt.axes(projection='3d')
#    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='viridis')
#    ax.plot_surface(X, Y, U, rstride=1, cstride=1, cmap='viridis')
    ax.plot_surface(X, Y, W, rstride=1, cstride=1, cmap='viridis')
    # TITLE: "Difference between numerical and analytical solution"
    ax.set_title('Mismunur tölulegu og fáguðu lausnarinnar')
    ax.set_xlabel("x axis")
    ax.set_ylabel("y axis")
    ax.set_zlabel("z axis")
    set_axes_equal(ax)
    plt.show()

def set_axes_equal(ax):
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)

    plot_radius = 0.5*max([x_range, y_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])

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
