import numpy as np
from math import *
import sympy
import scipy
import random

def regular_interval_matrix(Ac, Delta, make_prints = False, do_plot_orthants = False):
    """  
    """
    n = Ac.shape[0]
    orthants_visited = 0
    if np.linalg.matrix_rank(Ac) < n:
        S = Ac
        if make_prints:
            print("In regular_interval_matrix: Singularity of the midpoint matrix 'Ac'")
        return S, orthants_visited
    Ac_inv = np.linalg.inv(Ac)
    D = Delta @ np.abs(Ac_inv)
    dg = np.max(np.diag(D), axis=0)
    J = np.argmax(np.diag(D), axis=0)
    j = np.min(J)
    if dg >= 1:
        x = Ac_inv[:, j]
        S = vec2mat(Ac, Delta, x)
        if make_prints:
            print("In regular_interval_matrix: Singularity via diagonal condition")
        return S, orthants_visited
    if beeck_condition(Ac, Delta) < 1:
        S = []
        if make_prints:
            print("In regular_interval_matrix: Regularity via Beeck's condition")
        return S, orthants_visited
    S = singularity_det_descent(Ac, Delta)
    if len(S) != 0:
        if make_prints:
            print("In regular_interval_matrix: Singularity via steepest determinant descent")
        return S, orthants_visited
    AA = Ac.T @ Ac
    DD = Delta.T @ Delta
    if (np.linalg.matrix_rank(AA) == n) & (beeck_condition(AA, DD) < 1):
        S = []
        if make_prints:
            print("In regular_interval_matrix: Regularity  via symmetrization")
        return S, orthants_visited
    rs, S = regularity_suffcond_qz(Ac, Delta)
    if rs == 0:
        if make_prints:
            print("In regular_interval_matrix: Singularity as a by-product of 'regularity via two Qz-matrices'")
        return S, orthants_visited
    if rs == 1:
        S = []
        if make_prints:
            print("In regular_interval_matrix: Regularity via two Qz-matrices")
        return S, orthants_visited
    eps = 1e-7
    bc = Jansson_heuristic(Ac_inv)
    ep = np.max([n, 10]) * np.max([np.linalg.norm(Ac - Delta, ord=1), np.linalg.norm(Ac + Delta, ord=1), np.linalg.norm(bc, ord=1)]) * eps
    Z = [np.sign(Ac_inv @ bc)]
    D = []
    if do_plot_orthants:
        yy = []
        yy.append(0)
        yy.append(1)
        ii = 2
    while len(Z) != 0:
        p = len(Z)
        z = Z[-1]
        Z = Z[:-1]
        D.append(z)
        Q, S = qz_matrix(Ac, Delta, z)
        if len(S) != 0:
            orthants_visited = len(D)
            if make_prints:
                print("In regular_interval_matrix: Singularity via the main algorithm")
            if do_plot_orthants:
                to_plot(yy)
            return S, orthants_visited
        xut = Q @ bc
        if (xut[z==1] >= -ep).all():
            Q, S = qz_matrix(Ac, Delta, -z)
            if len(S) != 0:
                orthants_visited = len(D)
                if make_prints:
                    print("In regular_interval_matrix: Singularity via the main algorithm")
                if do_plot_orthants:
                    to_plot(yy)
                return S, orthants_visited
            xlt = Q @ bc
            if (xlt <= xut).all():
                for j in range(n):
                    zt = z
                    zt[j]= -zt[j]
                    if (xlt[j] * xut[j] <= ep) & (zt not in Z) & (zt not in D):
                        Z.append(zt)
            if do_plot_orthants:
                ii += 1
                yy[ii] = len(Z)
            if len(Z) + len(D) >= 20 * n**2:
                if do_plot_orthants:
                    to_plot(yy)
                S = []
                orthants_visited = []
                if make_prints:
                    print("In regular_interval_matrix: PROBLEM -- Program run has been stopped after reaching prescribed number of iterations")
    return S, orthants_visited
    S = []
    orthants_visited = len(D)
    if make_prints:
        print("In regular_interval_matrix: Regularity via the main algorithm")
    if do_plot_orthants:
        to_plot(yy)        
    return S, orthants_visited

def qz_matrix(Ac, Delta, z):
    n = Ac.shape[0]
    I = np.eye(n)
    Q = np.zeros((n,n))
    S = []
    for i in range(n):
        x, S, iterr = abs_value_eq(Ac.T, -np.diag(z.reshape(1,n)[0]) @ Delta.T, I[:,i])
        if len(S) != 0:
            Q = []
            S = S.T
            return Q, S
        Q[i,:] = x.T
    return Q, S

def abs_value_eq(A, B, b):
    b = b[:]
    n = len(b)
    I = np.eye(n)
    eps = 1e-7
    ep = n * (np.max([np.linalg.norm(A, ord=np.inf), np.linalg.norm(B, ord=np.inf), np.linalg.norm(b, ord=np.inf)])) * eps
    x = []
    S = []
    nbr_iter = 0
    if np.linalg.matrix_rank(A) < n:
        S = A
        return x, S, nbr_iter
    x = np.linalg.inv(A) @ b
    z = np.sign(x)
    ABdiagz = A + B @ np.diag(z)
    ABdiagz_inv = np.linalg.inv(ABdiagz)
    if np.linalg.matrix_rank(ABdiagz) < n:
        S = ABdiagz
        x = []
        return x, S, nbr_iter
    x = ABdiagz_inv.dot(b)
    C = - ABdiagz_inv @ B
    X = np.zeros((n,n))
    r = np.zeros(n)
    while (z.dot(x) < - ep).any():
        k = np.where(z.dot(x) < - ep)[0][0]
        nbr_iter += 1
        if 1 + 2*z[k]*C[k,k] <= 0:
            S = A + B @ ( np.diag(z) + (1/C[k,k]) * I[:,k].dot(I[k,:]) )
            x = []
            return x, S, nbr_iter
        if len(r[k+1:])==0:
            max_rk = 1e10
        else:
            max_rk = np.max(r[k+1:])
        if ( (k < n-1) & (r[k] > max_rk) ) | ( (k == n-1) & (r[k] > 0) ):
            x = x - X[:, k]
            z = np.sign(x)
            ct = A.dot(x)
            jm = np.abs(B).dot(np.abs(x))
            y = np.zeros(n)
            for i in range(n):
                if jm[i] > ep:
                    y[i] = ct[i]/jm[i]
                else:
                    y[i] = 1
            S = A - np.diag(y) @ np.abs(B) @ np.diag(z)
            x = []
            return x, S, nbr_iter
        X[:,k] = x
        r[k] = nbr_iter
        z[k] = -z[k]
        alpha = 2*z[k]/(1-2*z[k]*C[k,k])
        x = x + alpha * x[k] * C[:,k]
        C = C + alpha * C[:,k].dot(C[k,:])        
    return x, S, nbr_iter

def vec2mat(Ac, Delta, x):
    n = len(x)
    eps = 1e-7
    ep = np.max([n, 100]) * np.max([np.linalg.norm(Ac - Delta, ord=np.inf), np.linalg.norm(Ac + Delta, ord=np.inf), np.linalg.norm(x, ord=np.inf)]) * eps
    ct = Ac.dot(x)
    jm = Delta.dot(np.abs(x))
    y = np.zeros((n,1))
    z = y
    for i in range(n):
        if jm[i] > ep:
            y[i] = ct[i]/jm[i]
        else:
            y[i] = 1     
        if x[i] >= 0:
            z[i] = 1
        else:
            z[i] = -1    
    S = Ac - np.diag(y.reshape(1,n)[0]) @ Delta @ np.diag(z.reshape(1,n)[0])
    return S   

def regularity_suffcond_qz(Ac, Delta):
    n = Ac.shape[0]
    I = np.eye(n)
    e = np.ones((n,1))
    o = np.zeros((n,1))
    eps = 1e-7
    ep = np.max([n,10]) * np.max([np.linalg.norm(Ac - Delta, ord=1), np.linalg.norm(Ac + Delta, ord=1)]) * eps
    Ac_inv = np.linalg.inv(Ac)
    Q1, S = qz_matrix(Ac, Delta, -e)
    if len(S) != 0:
        rs = 0
        return rs, S
    Q2, S = qz_matrix(Ac, Delta, e)
    if len(S) != 0:
        rs = 0
        return rs, S
    A1 = np.concatenate((-Q1, e), axis=1)
    A2 = np.concatenate((-Ac_inv, e), axis=1)
    A3 = np.concatenate((I, o), axis=1)
    A4 = np.concatenate((-I, o), axis=1)
    A = np.concatenate((A1, A2, A3, A4), axis=0)
    b = o
    b = np.append(b, o)
    b = np.append(b, e)
    b = np.append(b, e)
    c = np.append(o, 1)
    x  = scipy.optimize.linprog(-c, A_ub=A, b_ub=b).x # min - c.dot(x) = max c.dot(x)
    if isinf(x[0]):
        rs = -1
        S = []
        return rs, S
    if x[n] > ep:
        rs = 1
        S = []
        return rs, S
    rs = -1
    S = S
    return rs, S

def singularity_det_descent(Ac, Delta):
    n = Ac.shape[0]
    Ad = Ac - Delta
    Ah = Ac + Delta
    if np.linalg.matrix_rank(Ac) < n:
        S = Ac
        return S
    if np.linalg.matrix_rank(Ah) < n:
        S = Ah
        return S
    A = Ah
    C = np.linalg.inv(A)
    beta = 0.5
    p = np.zeros((n,1))
    avoid_pb = 0
    while (beta < 0.95) & (avoid_pb < 1000):
        avoid_pb += 1
        Zd = (C.T >= 0)
        Zh = (C.T < 0)
        B = Zd * Ad + Zh * Ah
        beta = np.min(np.diag(B @ C), axis=0)
        #print("beta",beta)
        J = np.argmin(np.diag(B @ C), axis=0)
        k = np.min(J)
        if np.abs(beta) >= 0.95:
            S = []
            return S
        if (1e-10 < beta) & (beta < 0.95):
            D = B - A
            C = C - (1./beta) * C[:,k] @ (D[k,:] @ C)
            #C = np.linalg.inv(A)
            A[k,:] = B[k,:]
        if (0 < beta) & (beta <= 1e-10):
            S = []
            return S
        if beta <= 0:
            for i in range(n):
                p[i] = B[k,:i].dot(C[:i,k]) + A[k, i+1:].dot(C[i+1:,k])
            if (p > 0).all():
                S = []
                return S
            m = np.where(p <= 0)[0][0]
            if np.abs(C[m,k]) <= 1e-10:
                S = []
                return S
            A[k,:m-1] = B[k,:m-1]
            A[k,m] = - (B[k,:m-1].dot(C[:m-1,k]) + A[k,m+1:].dot(C[m+1:,k])) / C[m,k]
            S = A
            return S
    if avoid_pb == 1000:
        print("In singularity_det_descent: PROBLEM -- Program run has been stopped after reaching prescribed number of iterations when looking for beta")
    S = []
    return S

def to_plot(yy):
    qq = len(yy)
    xx = np.arange(qq) + 1
    plt.plot(xx,yy)
    plt.show();
    
def Jansson_heuristic(Ac_inv):
    n = Ac_inv.shape[0]
    bc = np.ones((n,1))
    g = np.min(np.abs(Ac_inv.dot(bc)))
    for i in range(n):
        for j in range(n):
            bp = bc
            bp[j] = -bp[j]
            if np.min(np.abs(Ac_inv @ bp)) > g:
                g = np.min(np.abs(Ac_inv @ bp))
                bc = bp
    for i in range(n):
        for j in range(n):
            if j != i:
                bp = bc
                bp[i] = -bp[i]
                bp[j] = -bp[j]
                if np.min(np.abs(Ac_inv @ bp)) > g:
                    g = np.min(np.abs(Ac_inv @ bp))
                    bc = bp
    return bc

def beeck_condition(Ac, Delta):
    return np.max(np.abs(np.linalg.eigvals(np.abs(np.linalg.inv(Ac)) @ Delta)))

def p_matrix_Rohn_v2(A, make_prints = False):
    """
    pm = 1 if A a P-matrix, 
    pm = 0 if not, 
    pm = -1 no answer.
    """
    n = A.shape[0]
    I = np.eye(n)
    if (np.linalg.det(A) < 0) | (np.diag(A) < 0).any(): # added for robustness
        if make_prints:
            print("p_matrix_Rohn_v2: Negative det of diagonal")
        return 0
    else:
        if np.linalg.matrix_rank(A-I) < n:
            if make_prints:
                print("p_matrix_Rohn_v2: Algorithm failed")
            return -1
        else:
            B = np.linalg.inv(A-I) @ (A+I)
            S, nbr_orthants_visited = regular_interval_matrix(B, I, make_prints, False)
            if len(S) != 0:
                if make_prints:
                    print("p_matrix_Rohn_v2: Not a P-matrix because S singular")
                return 0
            else:
                if make_prints:
                    print("p_matrix_Rohn_v2: P-matrix since no S singular")
                return 1