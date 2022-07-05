import numpy as np
from math import *
import sympy
import scipy
import random
import seaborn as sns
from matplotlib import pyplot as plt

def is_singular(matrix):
    assert matrix.shape[0] == matrix.shape[1]
    if np.linalg.det(matrix) == 0:
        return True
    else:
        return False
    
def is_sym_pos_def(A):
    if np.allclose(A, A.T):
        try:
            np.linalg.cholesky(A)
            return True
        except np.linalg.LinAlgError:
            return False
    else:
        return False
    
def generate_p_matrix(n):
    """
    n : square matrix dimension
    references : (PAGE 5) https://invenio.nusl.cz/record/81055/files/content.csg.pdf
                 (THEOREM 2) http://uivtx.cs.cas.cz/~rohn/publist/genpmat.pdf
    """
    C = 2*np.random.uniform(size=(n,n))-1
    C_inv = np.linalg.inv(C)
    D = np.random.uniform(size=(n,n))
    alpha = 0.95 / spectral_radius(np.abs(C_inv) @ D)
    return np.linalg.inv(C - alpha*D) @ (C + alpha*D)
    
def spectral_radius(A):
    return np.max(np.abs(np.linalg.eig(A)[0]))
        
def symmetrize(a):
    return a + a.T - np.diag(a.diagonal())

def get_wigner(size):
    a = np.random.normal(loc=0.0, scale=1.0, size=size).round(3)
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