# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 13:15:59 2021

@author: Notandi
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import spsolve

# Program 8.6 Finite element solver for 2D PDE
# with Dirichlet boundary conditions on a rectangle
# Input: rectangle domain [xl,xr]x[yb,yt] with MxN space steps
# Output: matrix w holding solution values
# Example usage: w=poissonfem(0,1,1,2,4,4)
def varmajafnvaegi(h, xl, xr, yb, yt, g1, g2, g3, g4, r, f):
    N = int((xr-xl)/h); M = int((yt-yb)/h); 
    m = M+1; n = N+1; mn = m*n
    k = h; h2 = h**2; k2 = k**2; hk = h*k
    x = np.linspace(xl, xr, n)
    y = np.linspace(yb, yt, m)
    A = np.zeros((mn,mn))
    b = np.zeros(mn)
    for i in range(1,m-1): # interior points
        for j in range(1,m-1):
            rsum=r(x[i]-2*h/3,y[j]-k/3)+r(x[i]-h/3,y[j]-2*k/3)
            +r(x[i]+h/3,y[j]-k/3)
            rsum=rsum+r(x[i]+2*h/3,y[j]+k/3)+r(x[i]+h/3,y[j]+2*k/3)
            +r(x[i]-h/3,y[j]+k/3)
            A[i+j*m,i+j*m]=2*(h2+k2)/(hk)-hk*rsum/18
            A[i+j*m,i-1+j*m]=-k/h-hk*(r(x[i]-h/3,y[j]+k/3)
            +r(x[i]-2*h/3,y[j]-k/3))/18
            A[i+j*m,i-1+(j-1)*m]=-hk*(r(x[i]-2*h/3,y[j]-k/3)
            +r(x[i]-h/3,y[j]-2*k/3))/18
            A[i+j*m,i+(j-1)*m]=-h/k-hk*(r(x[i]-h/3,y[j]-2*k/3)
            +r(x[i]+h/3,y[j]-k/3))/18
            A[i+j*m,i+1+j*m]=-k/h-hk*(r(x[i]+h/3,y[j]-k/3)
            +r(x[i]+2*h/3,y[j]+k/3))/18
            A[i+j*m,i+1+(j+1)*m]=-hk*(r(x[i]+2*h/3,y[j]+k/3)
            +r(x[i]+h/3,y[j]+2*k/3))/18
            A[i+j*m,i+(j+1)*m]=-h/k-hk*(r(x[i]+h/3,y[j]+2*k/3)
            +r(x[i]-h/3,y[j]+k/3))/18
            fsum=f(x[i]-2*h/3,y[j]-k/3)+f(x[i]-h/3,y[j]-2*k/3)
            +f(x[i]+h/3,y[j]-k/3)
            fsum=fsum+f(x[i]+2*h/3,y[j]+k/3)+f(x[i]+h/3,y[j]+2*k/3)
            +f(x[i]-h/3,y[j]+k/3)
            b[i+j*m]=-h*k*fsum/6
    for i in range(m):      # bottom and top boundary points
        j=0
        A[i+j*m,i+j*m]=1
        b[i+j*m]=g1(x[i])
        j=n-1
        A[i+j*m,i+j*m]=1
        b[i+j*m]=g2(x[i])
    for j in range(1,n-1):  # left and right boundary points
        i=0
        A[i+j*m,i+j*m]=1
        b[i+j*m]=g3(y[j])
        i=m-1
        A[i+j*m,i+j*m]=1
        b[i+j*m]=g4(y[j])
    c = spsolve(A,b)        # solve for solution in u labeling
    w = c.reshape(m, n)     # translate from u to w
    X,Y = np.meshgrid(x,y)
    ax = plt.axes(projection='3d')
    ax.plot_surface(X, Y, w, rstride=1, cstride=1,
                    cmap='viridis', edgecolor='none')
    ax.set_title('surface');
    return w

xl = 0; xr = 1; yb = 1; yt = 2 
h = 0.1
g1 = lambda x: np.log(x**2+1)   # define boundary values on bottom
g2 = lambda x: np.log(x**2+4)   # top
g3 = lambda y: 2*np.log(y)      # left side
g4 = lambda y: np.log(y**2+1)   # right side
r  = lambda x,y: 0.0*x+0.0*y
f  = lambda x,y: 0.0*x+0.0*y    # define input function data

varmajafnvaegi(h, xl, xr, yb, yt, g1, g2, g3, g4, r, f)