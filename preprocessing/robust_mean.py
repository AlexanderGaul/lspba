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