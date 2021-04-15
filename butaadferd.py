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
# with Dirichlet boundary conditions on a rectangle
# Input: rectangle domain [xl,xr]x[yb,yt] with MxN space steps
# Output: matrix w holding solution values
# Example usage: w=poissonfem(0,1,1,2,4,4)
def varmajafnvaegi(h, a, b, beta1, beta2, phi1, phi2):
    N = int(a/h); M = int(b/h); 
    n = N+1; m = M+1; mn = m*n
    x = np.linspace(0, b, n)
    y = np.linspace(0, b, m)
    A = np.zeros((mn,mn))
    b = np.zeros(mn)
    for i in range(1,n-1): # interior points
        for j in range(1,m-1):
            A[i+j*n,i+j*n]=4
            A[i+j*n,i-1+j*n]=-1
            A[i+j*n,i+(j-1)*n]=-1
            A[i+j*n,i+1+j*n]=-1
            A[i+j*n,i+(j+1)*n]=-1
            b[i+j*n]=0
    for i in range(n):      # bottom and top boundary points
        j=0
        A[i+j*n,i+j*n]=1
        b[i+j*n]=phi1(x[i])
        j=m-1
        A[i+j*n,i+j*n]=1
        b[i+j*n]=phi2(x[i])
    for j in range(1,m-1):  # left and right boundary points
        i=0
        A[i+j*n,i+j*n]=2
        A[i+j*n,i+j*n+1]=-1
        A[i+j*n,i+(j-1)*n]=-1/2
        A[i+j*n,i+(j+1)*n]=-1/2
        b[i+j*n]=0
        i=n-1
        A[i+j*n,i+j*n]=2
        A[i+j*n,i+j*n-1]=-1
        A[i+j*n,i+(j-1)*n]=-1/2
        A[i+j*n,i+(j+1)*n]=-1/2
        b[i+j*n]=0
    c = np.linalg.solve(A,b)    # solve for solution in u labeling
    V = c.reshape(m,n)         # translate from u to w
    X,Y = np.meshgrid(x,y)
    return X, Y, V

def phi_boundary(a, beta1, beta2):
    phi1 = lambda x: beta1*(np.sin(2*np.pi/a*(x-1/2))+1)        # bottom
    phi2 = lambda x: beta2*(np.cos(np.pi/4*(x-1))-1/2**(1/2))   # top
    return phi1, phi2

def plot_surf(X, Y, Z):
    ax = plt.axes(projection='3d')
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='plasma')
    ax.set_title('surface')
    ax.set_xlabel("x axis")
    ax.set_ylabel("y axis")
    ax.set_zlabel("z axis")
    plt.show()
    
def plot_cont(X, Y, Z):
    fig,ax=plt.subplots(1,1)
    cp = ax.contourf(X, Y, Z, cmap='plasma')
    fig.colorbar(cp)            # Add a colorbar to a plot
    ax.set_title('Filled Contours Plot')
    ax.set_xlabel('x axis')
    ax.set_ylabel('y axis')
    plt.show()

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