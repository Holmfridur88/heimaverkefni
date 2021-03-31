# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 13:15:24 2021

@author: Notandi
"""

# import math
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
# from bvp_ode import *
import matplotlib.pyplot as plt

u = lambda x: 0.5 * (2.0 + np.exp(2.0 - x) - np.exp(x))

def gildi(lamb):
    p = lambda x: 0.0 * x + 1.0
    q = lambda x: 0.0 * x - lamb ** 2
    f = lambda x: 0.0 * x

    a0 = q
    a1 = lambda x: 0.0 * x
    a2 = lambda x: 0.0 * x - 1.0
    
    return [p, q, f, a0, a1, a2]

lamb = 1/100
[p, q, f, a0, a1, a2] = gildi(lamb)
print(q(1))

