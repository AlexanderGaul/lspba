import numpy as np

def rho(s, tau=0.5) :
    return s / (s + tau**2)

def rho_prime(s, tau=0.5) :
    return tau**2 / (s + tau**2)**2

def rho_primeprime(s, tau=0.5) :
    return - 2 * tau**2 / (s + tau**2)**3

# Loss for scipy.optimize.least_squares 
def loss(s, tau=0.5) :
    return np.array([rho(s, tau),
                     rho_prime(s, tau),
                     rho_primeprime(s, tau)])

def f(x, patches, visibility_idx) :
    residuals = np.linalg.norm(x.reshape(-1, 16)[visibility_idx[:, 0], :] - patches, axis=1)
    return residuals

def jac(x, patches, visibility_idx) :
    vals = x.reshape(-1,16)[visibility_idx[:, 0], :] / np.linalg.norm(x.reshape(-1, 16)[visibility_idx[:, 0], :] - patches, axis=1).reshape(-1, 1)
    j = scipy.sparse.csr_matrix((vals.reshape(-1), (row_idx, col_idx)))
    return j

# TODO: optimize function for robust mean