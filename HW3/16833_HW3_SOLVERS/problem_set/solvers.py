'''
    Initially written by Ming Hsiao in MATLAB
    Rewritten in Python by Wei Dong (weidong@andrew.cmu.edu), 2021
'''

from scipy.sparse import csc_matrix, eye, csr_matrix
from scipy.sparse.linalg import inv, splu, spsolve, spsolve_triangular
from sparseqr import rz, permutation_vector_to_matrix, solve as qrsolve
import numpy as np
import matplotlib.pyplot as plt


def solve_default(A, b):
    from scipy.sparse.linalg import spsolve
    x = spsolve(A.T @ A, A.T @ b)
    return x, None


def solve_pinv(A, b):
    # TODO: return x s.t. Ax = b using pseudo inverse.
    N = A.shape[1]
    x = np.zeros((N, ))

    x = inv(A.T @ A) @ (A.T @ b)
    return x, None


def solve_lu(A, b):
    # TODO: return x, U s.t. Ax = b, and A = LU with LU decomposition.
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.splu.html
    N = A.shape[1]
    x = np.zeros((N, ))
    U = eye(N)

    A1 = A.T @ A
    lu = splu(A1, permc_spec = 'NATURAL')
    x = lu.solve(A.T @ b)
    U = lu.U
    return x, U


def solve_lu_colamd(A, b):
    # TODO: return x, U s.t. Ax = b, and Permutation_rows A Permutration_cols = LU with reordered LU decomposition.
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.splu.html
    N = A.shape[1]
    x = np.zeros((N, ))
    U = eye(N)

    A1 = A.T @ A
    lu = splu(A1, permc_spec = 'COLAMD')
    x = lu.solve(A.T @ b)
    U = lu.U

    # x = inv(U) @ inv(L) @ A.T @ b

    return x, U
def solve_lu_bonus(A,b):
    N = A.shape[1]
    x = np.zeros((N, ))
    U = eye(N)

    A1 = A.T @ A
    lu = splu(A1, permc_spec = 'COLAMD')
    # x = lu.solve(A.T @ b)
    U = lu.U
    L = lu.L

    # print("U shape = ", U.shape)
    # print("L shape = ", L.shape)
    Pr = csc_matrix((np.ones(N), (lu.perm_r, np.arange(N))))
    Pc = csc_matrix((np.ones(N), (np.arange(N), lu.perm_c)))

    B = Pr @ A.T @ b
    y = np.zeros((N, ))
    
    #forward
    y = spsolve_triangular(csr_matrix(L), B)
    y = U @ Pc @ inv(U) @ y
    
    #backward
    x = spsolve_triangular(csr_matrix(U), y, lower = False)    
    

    #forward
    # y[0] = B[0]/L[0,0]
    # for i in range(1,n):
    #     sum = 0
    #     for k in range(i-1):
    #         sum = sum + L[i,k] * y[k]
    #     y[i] = (B[i] - sum)/L[i,i]

    #backward
    # x = np.zeros((n,h))
    # U = inv((U@Pc.T@inv(U))) @U @ Pc.T
    # x[n-1] = y[n-1] / U[n-1, n-1]
    # for i in range(n-2, 0, -1):
    #     sum = 0
    #     for k in range(i, n-1):
    #         sum = sum + U[i, k] * x[k]
    #     x[i] = (y[i] - sum)/U[i,i]


    return x, U


def solve_qr(A, b):
    # TODO: return x, R s.t. Ax = b, and |Ax - b|^2 = |Rx - d|^2 + |e|^2
    # https://github.com/theNded/PySPQR
    N = A.shape[1]
    x = np.zeros((N, ))
    R = eye(N)

    z, R, E, rank = rz(A, b, tolerance = None, permc_spec='NATURAL')

    x = spsolve_triangular(csr_matrix(R), z, lower=False)
    return x, R


def solve_qr_colamd(A, b):
    # TODO: return x, R s.t. Ax = b, and |Ax - b|^2 = |R E^T x - d|^2 + |e|^2, with reordered QR decomposition (E is the permutation matrix).
    # https://github.com/theNded/PySPQR
    N = A.shape[1]
    x = np.zeros((N, ))
    R = eye(N)

    z, R, E, rank = rz(A, b, tolerance = None, permc_spec='COLAMD')
    x = spsolve_triangular(csr_matrix(R), z, lower=False)
    x = permutation_vector_to_matrix(E) @ x
    return x, R


def solve(A, b, method='default'):
    '''
    \param A (M, N) Jacobian matirx
    \param b (M, 1) residual vector
    \return x (N, 1) state vector obtained by solving Ax = b.
    '''
    M, N = A.shape

    fn_map = {
        'default': solve_default,
        'pinv': solve_pinv,
        'lu': solve_lu,
        'qr': solve_qr,
        'lu_colamd': solve_lu_colamd,
        'qr_colamd': solve_qr_colamd,
        'lu_bonus' : solve_lu_bonus,
    }

    return fn_map[method](A, b)
