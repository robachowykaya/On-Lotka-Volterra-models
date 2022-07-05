import numpy as np
from math import *
import sympy
import scipy
import random

def p_matrix_Rohn_v1(A, make_prints = False):
    """
    A : square matrix
    complexity of the algorithm : not exponential, "finite number of steps" (THEOREM 3) https://invenio.nusl.cz/record/81055/files/content.csg.pdf
    references : https://invenio.nusl.cz/record/81055/files/content.csg.pdf
                 http://uivtx.cs.cas.cz/~rohn/matlab/
                 (NOTATIONS) https://www.researchgate.net/publication/228571326_An_Algorithm_for_Checking_Regularity_of_Interval_Matrices
                 https://www.sciencedirect.com/science/article/pii/S0024379511001418
                 (THEOREMS BASIS IN) https://www.sciencedirect.com/science/article/pii/S0024379501005900
    Check the P-property, returns
    pm = 1 if A a P-matrix, 
    pm = 0 if not (and then J is the index of one problematic principal minor), 
    pm = -1 no answer.
    """
    if (np.linalg.det(A) < 0) | (np.diag(A) < 0).any(): # added for robustness
        return 0, []
    pm = 1
    J = []
    n = A.shape[0]
    e = np.ones(n)
    I = np.eye(n)
    if is_sym_pos_def(A):
        if make_prints:
            print("In p_matrix_Rohn_v1: is_sym_pos_def(A)")
        return pm, J
    if is_singular(A - I) | is_singular(A + I):
        pm = -1
        if make_prints:
            print("In p_matrix_Rohn_v1: is_singular(A - I) | is_singular(A + I)")
        return pm, J
    C = np.linalg.inv(A - I) @ (A + I)
    R = np.linalg.inv(C)
    if spectral_radius(np.abs(R)) < 1:
        if make_prints:
            print("In p_matrix_Rohn_v1: spectral_radius(np.abs(R)) < 1")
        return pm, J
    B = [C - I, C + I] # interval matrix
    b = e
    gamma = np.min(np.abs(R.dot(b)))
    for i in range(n):
        for j in range(n):
            new_b = b
            new_b[j] = - new_b[j]
            if np.min(np.abs(R.dot(new_b))) > gamma:
                gamma = np.min(np.abs(R.dot(new_b)))
                b = new_b
    x, S = intervallhull(B, [b - np.ones(n),b + np.ones(n)])
    # x the solution set OR S a singular matrix in the interval matrix
    if len(x) != 0:
        if make_prints:
            print("In p_matrix_Rohn_v1: intervallhull")
        return pm, J
    # the following does not really work (purpose: identify a negative prinicipal minor):
    if is_singular(S): # Sx 
        x = sympy.symbols([f"x{idx}" for idx in range(S.shape[1])])
        gen_sol = sympy.solve(S.dot(np.array(x)), *x, particular=True, quick=True)
        x = np.array([gen_sol[key] for key in x]) # solve Sx = 0 even for singular matrix S
    else:
        x = np.linalg.solve(S, np.zeros(S.shape[1])) # should be 0
    if (x != 0).all():
        y = np.ones(n)
        y[x!=0] = C.dot(x)[x!=0] / x[x!=0]
        for i in range(n):
            if (y[i] != -1) & (y[i] != 1):
                y[i] = 1
                if np.linalg.det(A - I) * np.linalg.det(C - np.diag(y)) > 0:
                    y[i] = -1    
        pm = 0
        J = np.where(y == -1)[0]
    else:
        pm = 0
        J = []
    if make_prints:
        print("In p_matrix_Rohn_v1: determines minor negative")
    return pm, J

def intervallhull(A, b):
    """
    A : interval matrix of the form [minimal matrix, maximal matrix]
    b : interval vector
    Computes either the interval hull x of the solution set of Ax = b (for any matrix A and vector b in the intervals) 
    OR a singular matrix S in the interval matrix A.
    """
    n = A[0].shape[0]
    x = []
    S = []
    Ac = 0.5 * (A[0] + A[1]) # center matrix
    bc = 0.5 * (b[0] + b[1]) # center vector
    delta = 0.5 * (b[1] - bc) # radius vector
    if is_singular(Ac):
        S = Ac
        return x, S
    xc = np.linalg.inv(Ac).dot(bc)
    z = np.sign(xc)
    x_floor = xc
    x_ceilling = xc
    Z = []
    Z.append(list(z))
    D = []
    while len(Z) != 0:
        z = list(random.sample(Z, 1)[0])
        if z in Z:
            Z.remove(z)
            D.append(z)
        Qz, S = qzmatrix(A, np.array(z))
        if len(S)!=0:
            x = []
            return x, S
        Qminusz, S = qzmatrix(A, -np.array(z))
        if len(S)!=0:
            x = []
            return x, S
        xz_ceilling = Qz.dot(bc) + np.abs(Qz).dot(delta)
        xz_floor = Qminusz.dot(bc) - np.abs(Qminusz).dot(delta)
        if (xz_floor <= xz_ceilling).all():
            x_floor = np.minimum(x_floor, xz_floor)
            x_ceilling = np.maximum(x_ceilling, xz_ceilling)  
            for j in range(n):
                new_z = z
                new_z[j] = -new_z[j]

                if (xz_floor[j]*xz_ceilling[j] <= 0) & (new_z not in Z) & (new_z not in D):
                    Z.append(new_z)
    x = [x_floor, x_ceilling]
    S = []
    return x, S

def qzmatrix(A, z):
    """
    A : interval matrix of the form [minimal matrix, maximal matrix]
    b : sign vector
    Computes either Q solution of Q Ac - |Q| Delta Tz = I 
    OR a singular matrix S.
    """
    n = A[0].shape[0]
    Ac = 0.5 * (A[0] + A[1]) # center matrix
    Delta = 0.5 * (A[1] - Ac) # radius matrix
    Tz = np.diag(z)
    Qz = np.zeros((n,n))
    for i in range(n):
        x, S = absvaleqn(Ac.T, - Tz @ Delta.T, np.eye(n)[:,i])
        if len(S)!= 0:
            S = S.T
            Qz = []
            return Qz, S
        Qz[i,:] = x.T
    S = []
    return Qz, S

def absvaleqn(A, B, b):
    """
    A : matrix
    B : matrix
    b : vector
    Computes either x solution of Ax + B|x| = b 
    OR a singular matrix S st. |S-A| <= |B|.
    """
    n = A.shape[0]
    x = []
    S = []
    i = 0
    r = np.zeros(n)
    X = np.zeros((n,n))
    if is_singular(A):
        S = A
        return x, S
    z = np.sign(np.linalg.inv(A).dot(b))
    Tz = np.diag(z)
    ABTz = A + B @ Tz
    ABTz_inv = np.linalg.inv(ABTz)
    if is_singular(ABTz):
        S = ABTz
        return x, S
    x = ABTz_inv.dot(b)
    C = - ABTz_inv @ B
    neg_xz_idx = np.where(z.dot(x) < 0)[0]
    while len(neg_xz_idx)!=0:
        i += 1
        k = neg_xz_idx[0]
        if 1 + 2*z[k]*C[k,k] <= 0:
            S = A + B @ (Tz + (1./C[k,k]) * np.eye(n)[:,k].reshape(-1,1).dot(np.eye(n)[:,k].reshape(1,-1)))
            x = []
            return x, S
        if len(r[k+1:])==0:
            max_rk = 1e10
        else:
            max_rk = np.max(r[k+1:])
        if ( (k < n-1) & (r[k] > max_rk) ) | ( (k == n-1) & (r[n-1] > 0) ):
            x = x - X[:,k]
            y = np.zeros(n)
            for j in range(n):
                if np.abs(B).dot(np.abs(x))[j] > 0:
                    y[j] = A.dot(x)[j] / np.abs(B).dot(np.abs(x))[j]
                else:
                    y[j] = 1
            z = np.sign(x)
            Ty = np.diag(y)
            S = A - Ty @ np.abs(B) @ Tz
            x = []
            return x, S
        r[k] = i
        X[:,k] = x
        z[k] = -z[k]
        alpha = 2*z[k] / (1 - 2*z[k]*C[k,k])
        x = x + alpha*x[k]*C[:,k]
        C = C + alpha * C[:,k].reshape(-1,1).dot(C[k,:].reshape(1,-1))
        neg_xz_idx = np.where(z*x < 0)[0]
    return x, S