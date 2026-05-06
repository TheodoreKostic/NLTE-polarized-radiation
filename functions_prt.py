import numpy as np
import matplotlib.pyplot as plt
from numba import jit

# ---------------------------------------------------
# Functions and constants for polarized radiative transfer
# ---------------------------------------------------

B = 1.0 # Planck function

def hanle_factor(Gamma):
    return 1/5 + (4/5)/(1 + Gamma**2)

def doppler_profile(x):
    return np.exp(-x**2) / np.sqrt(np.pi)

def init_tensor_1D(N):
    S = {}
    S[(0,0)] = np.ones(N) * B
    S[(2,0)] = np.zeros(N)

    return S

def init_tensor(N):
    S = {}
    S[(0,0)] = np.ones(N) * B
    S[(2,0)] = np.zeros(N)

    for q in ['1c','1s','2c','2s']:
        S[(2,q)] = np.zeros(N)

    return S

# --- Short Characteristics ---
@jit(nopython=True)
def short_characteristics(tau, S, mu, I_boundary, ali = False):
    ND = S.shape[0]
    begin = ND-1
    end = -1
    step = -1
    if mu < 0:
        begin = 0
        end = ND
        step = 1

    I = np.zeros(ND)
    L = np.zeros(ND)
    I[begin] = I_boundary

    for d in range(begin+step,end-step,step):
        delta_u = (tau[d-step] - tau[d])/mu
        delta_d = (tau[d] - tau[d+step])/mu
        expd = np.exp(-delta_u)

        if delta_u <= 0.01:
            du = delta_u
            w0 = du*(1.-du/2.+du**2/6.-du**3/24.+du**4/120.-du**5/720.+du**6/5040.-du**7/40320.+du**8/362880.)
            w1 = du**2*(0.5-du/3.+du**2/8.-du**3/30.+du**4/144.-du**5/840.+du**6/5760.-du**7/45360.+du**8/403200.)
            w2 = du**3*(1./3.-du/4.+du**2/10.-du**3/36.+du**4/168.-du**5/960.+du**6/6480.-du**7/50400.+du**8/443520.)
        else:
            w0 = 1.0 - expd
            w1 = w0 - delta_u * expd
            w2 = 2.0 * w1 - delta_u**2 * expd

        psi0 = w0 + (w1*(delta_u/delta_d - delta_d/delta_u) - w2*(1.0/delta_d + 1.0/delta_u))/(delta_u+delta_d)
        psiu = (w2/delta_u + w1*delta_d/delta_u)/(delta_u+delta_d)
        psid = (w2/delta_d - w1*delta_u/delta_d)/(delta_u+delta_d)

        I[d] = I[d-step]*expd + psiu*S[d-step] + psi0*S[d] + psid*S[d+step]
        L[d] = psi0

    # last point linear
    d = end-step
    delta_u = (tau[d-step]-tau[d])/mu
    expd = np.exp(-delta_u)
    if delta_u < 0.01:
        expd = 1.0 - delta_u + delta_u**2/2 - delta_u**3/6
        psi0 = delta_u/2 - delta_u**2/6 + delta_u**3/24
        psiu = delta_u/2 - delta_u**2/3 + delta_u**3/8
    else:
        psi0 = 1.0 - (1.0 - expd)/delta_u
        psiu = -expd + (1.0 - expd)/delta_u

    I[d] = I[d-step]*expd + psiu*S[d-step] + psi0*S[d]
    L[d] = psi0

    return I, L

# -----------------------
# TENSOR COMPUTATION
# -----------------------
def T20(mu):
    return 0.5 * (3*mu**2 - 1)

def T2Q(mu):
    return 1.5 * (1 - mu**2)

def emergent_stokes(S, mu):
    T_I = T20(mu)
    T_Q = T2Q(mu)

    I_out = S[(0,0)] + T_I * S[(2,0)]
    Q_out = T_Q * S[(2,0)]

    return I_out, Q_out

def compute_tensors(mu, chi):

    sin_t = np.sqrt(1 - mu**2)

    cos_chi = np.cos(chi)
    sin_chi = np.sin(chi)
    cos2 = np.cos(2*chi)
    sin2 = np.sin(2*chi)

    T = {}

    # Q=0
    T[('I',0)] = 0.5/np.sqrt(2)*(3*mu**2 - 1)
    T[('Q',0)] = 3/(2*np.sqrt(2))*(1 - mu**2)
    T[('U',0)] = 0.0

    # Q=1
    f1 = np.sqrt(3)/2 * mu * sin_t

    T[('I','1c')] = -f1*cos_chi
    T[('I','1s')] = -f1*sin_chi
    T[('Q','1c')] = T[('I','1c')]
    T[('Q','1s')] = T[('I','1s')]
    T[('U','1c')] = -np.sqrt(3)/2 * sin_t * sin_chi
    T[('U','1s')] =  np.sqrt(3)/2 * sin_t * cos_chi

    # Q=2
    f2 = np.sqrt(3)/4 * (1 - mu**2)

    T[('I','2c')] = f2*cos2
    T[('I','2s')] = f2*sin2
    T[('Q','2c')] = -np.sqrt(3)/4*(1 + mu**2)*cos2
    T[('Q','2s')] = -np.sqrt(3)/4*(1 + mu**2)*sin2
    T[('U','2c')] =  np.sqrt(3)/2 * mu * sin2
    T[('U','2s')] = -np.sqrt(3)/2 * mu * cos2

    return T

def hanle_matrix(Gamma, theta_B, chi_B):
    """
    Compute 5x5 Hanle matrix for quadrupole components.
    Gamma = B / (Gamma_rad * sqrt(1 + (Gamma_col/Gamma_rad)^2))  # dimensionless field strength
    theta_B, chi_B: magnetic field orientation angles
    Returns matrix H such that rho^2_q_new = H @ rho^2_q_old
    """
    # Simplified for weak field; full implementation needs rotation matrices
    # For now, use identity for zero field; expand for general case
    H = np.eye(5)  # Placeholder: implement full matrix from paper
    # TODO: Implement based on paper Eq. (28)-(30)
    return H
