import numpy as np
from math import *
import sympy
import random

def p_matrix_Tsatsomeros(A, tol = 1e-5):
    """
    A : square matrix
    
    complexity : O(2^n)
    
    REF :
    
        algo based on : (PAGE 110) https://reader.elsevier.com/reader/sd/pii/S0024379506002126?token=0E90031DA572FFFFF30471F25B729A527B43C13CCF2D36B83AD25F8467299E667AB1F2442E394749FFE5CCAA35A4DE57&originRegion=eu-west-1&originCreation=20220525080600
    
        also introduced in : (PAGE 22 & theorem 7.3) http://www.math.wsu.edu/faculty/tsat/files/PmatricesLectureNotes.pdf

        and : (PAGE 4) http://www.math.wsu.edu/faculty/tsat/files/tl_c.pdf
    
    """
    n = len(A)
    if A[0,0] <= 0:
        result = False
    elif n==1:
        result = True  
    else:
        B = A[1:,1:]
        D = A[1:,0] / A[0,0]
        C = B - D * A[0,1:]
        Im_C = np.imag(C)
        C = np.real(C) + 1j * (abs(Im_C)>tol) * Im_C
        result = p_matrix_Tsatsomeros(B) & p_matrix_Tsatsomeros(C)
    return result

def is_singular(matrix):
    assert matrix.shape[0] == matrix.shape[1]
    return not bool( np.linalg.det(matrix) )
    
def is_sym_pos_def(A):
    if np.array_equal(A, A.T):
        try:
            np.linalg.cholesky(A)
            return True
        except np.linalg.LinAlgError:
            return False
    else:
        return False
    
def generate_p_matrix(n):
    """
    REF :
    
        also based on : http://uivtx.cs.cas.cz/~rohn/publist/genpmat.pdf
    
        and explicitly presented in : (PAGE 5) https://invenio.nusl.cz/record/81055/files/content.csg.pdf
    
    """
    C = 2*np.random.uniform(size=(n,n))-1
    C_inv = np.linalg.inv(C)
    D = np.random.uniform(size=(n,n))
    alpha = 0.95 / spectral_radius(np.abs(C_inv) @ D)
    return np.linalg.inv(C - alpha*D) @ (C + alpha*D)
    
def spectral_radius(A):
    return np.max(np.abs(np.linalg.eig(A)[0]))
    
def p_matrix_Rohn(A):
    """
    A : square matrix
    
    complexity : not exponential
    
    REF :
    
        algo introduced in : https://invenio.nusl.cz/record/81055/files/content.csg.pdf (http://uivtx.cs.cas.cz/~rohn/matlab/)
    
        (some notations are not explained and can be found here : 
        - https://www.researchgate.net/publication/228571326_An_Algorithm_for_Checking_Regularity_of_Interval_Matrices
        - https://reader.elsevier.com/reader/sd/pii/S0024379511001418?token=13BC638F227811A5E0826300FE5E84A435070734D7C6BB750BBDDE6DDBB8975E7ED476F0AFBB55C3DCBFCC4FBB34EFAB&originRegion=eu-west-1&originCreation=20220525090251)
    
        motivated by : https://reader.elsevier.com/reader/sd/pii/S0024379501005900?token=324BA2ACA86AFA8E638235CAE692927FD3599FB79221CC9B115CBA136C28DAEBF89EAF78B5C80BF0F8214AE2475F1D24&originRegion=eu-west-1&originCreation=20220523153804
        
    Check the P-property, return pm = 1 if A a P-matrix, pm = 0 if not (and then J is the index of one problematic principal minor), pm = -1 no answer.
    
    """
    pm = 1
    J = []
    n = A.shape[0]
    e = np.ones(n)
    I = np.eye(n)
    
    if is_sym_pos_def(A):
        return pm, J
    
    if is_singular(A - I) | is_singular(A + I):
        pm = -1
        return pm, J
    
    C = np.linalg.inv(A - I) @ (A + I)
    R = np.linalg.inv(C)
    
    if spectral_radius(np.abs(R)) < 1:
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
                
    x, S = intervallhull(B, [b,b])
    
    if len(x) != 0:
        return pm, J
    
    # problem in soling the following in order to fullfill J...
    x = sympy.symbols([f"x{idx}" for idx in range(S.shape[1])])
    gen_sol = sympy.solve(S.dot(np.array(x)), *x, particular=True, quick=True)
    x = np.array([gen_sol[key] for key in x]) # solve Sx = 0 even for singular matrix S
    
    y = np.ones(n)
    y[x!=0] = C.dot(x)[x!=0] / x[x!=0]

    for i in range(n):
        if (y[i] != -1) & (y[i] != 1):
            y[i] = 1
            if np.linalg.det(A - I)*np.linalg.det(C - np.diag(y)) > 0:
                y[i] = -1
                
    pm = 0
    J = np.where(y == -1)[0]
    
    return pm, J

def intervallhull(A, b):
    """
    A : interval matrix of the form [minimal matrix, maximal matrix]
    b : interval vector
    
    Computes either the interval hull x of the solution set of Ax = b (for any matrix A and vector b in the intervals) OR a singular matrix S in the interval matrix A.
    """
    n = A[0].shape[0]
    x = []
    S = []
    delta = 1e-5*np.ones(n)
    Ac = 0.5 * (A[0] + A[1]) * np.ones((n,n))
    bc = 0.5 * (b[0] + b[1]) * np.ones(n)
    
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
        Qz, S = qzmatrix(A,np.array(z))
        
        if len(S)!=0:
            x = []
            return x, S
        
        Qminusz, S = qzmatrix(A, -np.array(z))
        
        if len(S)!=0:
            x = []
            return x, S
        
        xz_ceilling = Qz.dot(bc) + np.abs(Qz).dot(delta)
        xz_floor = Qminusz.dot(bc) - np.abs(Qminusz).dot(delta)
        
        if xz_floor.all() <= xz_ceilling.all():
            x_floor = np.minimum(x_floor, xz_floor)
            x_ceilling = np.maximum(x_ceilling, xz_ceilling)
            
            for j in range(n):
                new_z = z
                new_z[j] = -new_z[j]

                if (xz_floor[j]*xz_ceilling[j] <= 0) & (new_z not in Z) & (new_z not in D):
                    Z.append(new_z)

    x = [x_floor, x_ceilling]
    
    return x, S

def qzmatrix(A, z):
    """
    A : interval matrix of the form [minimal matrix, maximal matrix]
    b : sign vector
    
    Computes either Q solution of Q Ac - |Q| Delta Tz = I OR a singular matrix S.
    """
    n = A[0].shape[0]
    Ac = 0.5 * (A[0] + A[1]) * np.ones((n,n))
    e = np.ones(n)
    Delta = A[0] - Ac
    Tz = np.diag(z)
    
    for i in range(n):
        x, S = absvaleqn(Ac.T, - Tz @ Delta.T, np.eye(n)[i])
        
        if len(S)!= 0:
            S = S.T
            Qz = []
            return Qz, S
        Qz = np.zeros((n,n))
        Qz[i,:] = x.T
        
    S = []
    return Qz, S

def absvaleqn(A, B, b):
    """
    A : matrix
    B : matrix
    b : vector
    
    Computes either x solution of Ax + B|x| = b OR a singular matrix S st. |S-A| <= |B|.
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
    
    j_idx = np.where(z*x < 0)[0]
    neg_xz_idx = j_idx.copy()
    
    for j in j_idx:
        while z[j]*x[j] < 0:
            i += 1
            k = neg_xz_idx[0]
            neg_xz_idx = neg_xz_idx[1:]
            
            if 1 + 2*z[k]*C[k,k] <= 0:
                S = A + B @ (Tz + (1./C[k,k]) * np.eye(n)[k,:].dot(np.eye(n)[k,:].T))
                x = []
                return x, S
            
            if ( (k < n-1) & (r[k] > np.max(r[k+1:])) ) | ( (k == n-1) & (r[n-1] > 0) ):
                x = x - X[:,k]
                
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
            C = C + alpha* C[:,k].dot(C[:,k])
    return x, S
        
def symmetrize(a):
    return a + a.T - np.diag(a.diagonal())

def get_wigner(size):
    a = np.random.normal(loc=0.0, scale=1.0, size=size)
    a = np.triu(a)
    return symmetrize(a)

def plot_spectrum(n=100, alpha=1, title=True):
    B = np.eye(n) - (np.random.randn(n, n))*(1/(np.sqrt(n)*alpha))
    eig_B = np.linalg.eigvals(B)
    # The rest of the function is dedicated to the plot.
    radius = 1/alpha  # radius of the circle.
    t = np.linspace(0, 2*np.pi, 100)
    fig = plt.figure(1, figsize=(10, 6))
    plt.plot(1 - radius*np.cos(t), radius*np.sin(t), color='k')
    plt.plot(eig_B.real, eig_B.imag, '.', color='k', label = r"$\alpha$ = {0}".format(alpha))
    plt.axis("equal")
    plt.xlabel(r"Real axis", fontsize=15)
    plt.ylabel("Imaginary axis", fontsize=15)
    if title:
        plt.title(r'Circular law, $\alpha$ = {}, (N = {})'.format(alpha, n), fontsize=15)
    plt.show()
    plt.close()