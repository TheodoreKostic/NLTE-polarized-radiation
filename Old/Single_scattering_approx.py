import numpy as np
import matplotlib.pyplot as plt
from numba import jit

# Single-scattering approximation for polarized light scattering in a plane-parallel atmosphere

# Use S_I from the unpolarized case as the source function for the polarized case (1 iteration of the scattering process)
# From this S_I we compute I
# Then we compute J from I
# Finally we compute S_Q, S_U, S_V from J, after which we do formal solution of the polarized radiative transfer equation to get Q, U, V

# Define the optical depth grid
N_tau = 100
tau = np.logspace(-4, 4, N_tau)

# Define the angular grid
N_mu = 12   
mu, w_mu = np.polynomial.legendre.leggauss(N_mu)

# Define frequency grid 
N_nu = 41
x = np.linspace(-5, 5, N_nu)

# --- Doppler profile ---
phi = np.exp(-x**2) / np.sqrt(np.pi)
phi /= np.trapezoid(phi, x)

# --- Physical parameters ---
epsilon = 1e-4
B = 1.0  

# 1. Compute S_I in the unpolarized case (single scattering approximation)
