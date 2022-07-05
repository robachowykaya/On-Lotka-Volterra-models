from general_utils import *
import numpy as np
from math import *
import sympy
import scipy
import random

def p_matrix_Tsatsomeros(A, make_prints):
    """
    A : square matrix
    tol : small tolerance used to zero out small imaginary parts that show up in the Schur complement
    complexity of the algorithm : O(2^n)
    references : (PAGE 110) https://www.sciencedirect.com/science/article/pii/S0024379506002126
                 (PAGE 22 & THEOREM 7.3) http://www.math.wsu.edu/faculty/tsat/files/PmatricesLectureNotes.pdf
                 (PAGE 4) http://www.math.wsu.edu/faculty/tsat/files/tl_c.pdf
    """
    n = len(A)
    tol = 1e-5
    if (A[0,0] <= 0) | (np.iscomplex(A[0,0])):
        result = False
    elif n==1: 
        result = True 
    else:
        B = A[1:,1:]
        D = A[1:,0].reshape(-1,1) @ np.linalg.inv(np.array([[A[0,0]]]))
        C = B - D @ A[0,1:].reshape(1,-1)
        Im_C = np.imag(C)
        C = np.real(C) + 1j * (abs(Im_C)>tol) * Im_C
        result = p_matrix_Tsatsomeros(B, make_prints) & p_matrix_Tsatsomeros(C, make_prints)
    return result