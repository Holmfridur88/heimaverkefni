# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 13:15:59 2021

@author: Notandi
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import spsolve
from mpl_toolkits import mplot3d

# Program 8.6 Finite element solver for 2D PDE
# with von Neuman boundary conditions on a rectangle
# Input: rectangle domain [0,a]x[0,b] with NxM space steps
# Output: matrix V holding solution values
def varmajafnvaegi(h, a, b, beta1, beta2, phi1, phi2):
    N = int(a/h); M = int(b/h); 
    n = N+1; m = M+1; mn = m*n
    x = np.linspace(0, a, n)
    y = np.linspace(0, b, m)
    A = np.zeros((mn,mn))
    b = np.zeros(mn)
    for i in range(1,n-1): # interior points
        for j in range(1,m-1):
            A[i+j*n, i+j*n]     =  4
            A[i+j*n, i+j*n-1]   = -1
            A[i+j*n, i+j*n+1]   = -1
            A[i+j*n, i+(j-1)*n] = -1
            A[i+j*n, i+(j+1)*n] = -1
            b[i+j*n] = 0
    for i in range(n):      # bottom and top boundary points
        j = 0       # bottom
        A[i+j*n, i+j*n]     = 1
        b[i+j*n] = phi1(x[i])
        j = m-1     # top
        A[i+j*n, i+j*n]     = 1
        b[i+j*n] = phi2(x[i])
    for j in range(1,m-1):  # left and right boundary points
        i = 0       # left
        A[i+j*n, i+j*n]     =  2
        A[i+j*n, i+j*n+1]   = -1
        A[i+j*n, i+(j-1)*n] = -1/2
        A[i+j*n, i+(j+1)*n] = -1/2
        b[i+j*n] = 0
        i = n-1     # right
        A[i+j*n, i+j*n]     =  2
        A[i+j*n, i+j*n-1]   = -1
        A[i+j*n, i+(j-1)*n] = -1/2
        A[i+j*n, i+(j+1)*n] = -1/2
        b[i+j*n] = 0
    c = np.linalg.solve(A,b)    # solve for solution in c labeling
    V = c.reshape(m,n)          # projecting c to a grid
    X,Y = np.meshgrid(x,y)
    return X, Y, V

def phi_boundary(a, beta1, beta2):
    phi1 = lambda x: beta1*(np.sin(2*np.pi/a*(x-1/2))+1)        # bottom
    phi2 = lambda x: beta2*(np.cos(np.pi/4*(x-1))-1/2**(1/2))   # top
    return phi1, phi2

def plot_surf(X, Y, Z):
    ax = plt.axes(projection='3d')
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='coolwarm')
    ax.set_title('Hitastig sem fall af staðsetningu')
    ax.set_xlabel("x axis")
    ax.set_ylabel("y axis")
    ax.set_zlabel("z axis")
    set_axes_equal(ax)
    plt.show()
    
def plot_cont(X, Y, Z):
    fig,ax=plt.subplots(1,1)
    cp = ax.contourf(X, Y, Z, cmap='coolwarm')
    cb = fig.colorbar(cp, shrink=0.65)            # Add a colorbar to a plot
    cb.set_label('Hitastig')
    ax.set_title('Hitakort')
    ax.set_xlabel('x axis')
    ax.set_ylabel('y axis')
    ax.set_aspect('equal')
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
a = [1, 2]; b = 1; h = [1/4, 1/50]
beta1 = 1; beta2 = [0, 2]

# LIÐUR II
phi1_II, phi2_II = phi_boundary(a[0], beta1, beta2[0])
X, Y, HZ = varmajafnvaegi(h[0], a[0], b, beta1, beta2[0], phi1_II, phi2_II)
print(HZ)

# LIÐUR III
phi1_III, phi2_III = phi_boundary(a[1], beta1, beta2[1])    
X, Y, V = varmajafnvaegi(h[1], a[1], b, beta1, beta2[1], phi1_III, phi2_III)
plt.figure(1)
plot_surf(X, Y, V)
plot_cont(X, Y, V)

# LIÐUR IV
phi1_IV = lambda x: beta1*(np.sin(np.degrees(2*np.pi/a[1]*(x-1/2)))+1)          # bottom
phi2_IV = lambda x: beta2[1]*(np.cos(np.degrees(np.pi/4*(x-1)))-1/2**(1/2))     # top
X, Y, Z = varmajafnvaegi(h[1], a[1], b, beta1, beta2[1], phi1_IV, phi2_IV)
plt.figure(3)
plot_surf(X, Y, Z)
