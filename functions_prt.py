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

    return np.stack((I, L))

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

def hanle_matrix_magnetic_frame(Gamma_rad, Gamma_col, omega_L):
    """
    Full Hanle matrix in MAGNETIC FRAME (irreducible tensor formalism).
    
    In the magnetic frame, the matrix is diagonal with Q-dependent depolarization rates.
    
    Parameters:
    -----------
    Gamma_rad : float
        Natural decay rate (A_ul)
    Gamma_col : float
        Collisional broadening rate
    omega_L : float
        Larmor precession frequency = g * mu_B * B / hbar
    
    Returns:
    --------
    H : 5x5 complex array
        Hanle matrix in magnetic frame [Q=-2, -1, 0, 1, 2]
        Acts on density matrix components: rho^2_q_new = H @ rho^2_q_old
    
    Notes:
    ------
    In the magnetic frame, the statistical equilibrium equation is:
        dρ^K_Q/dt = -[Γ_total + i·Q·ω_L]·ρ^K_Q
    
    where:
    - Γ_total = Gamma_rad + Gamma_col (total damping)
    - ω_L is the Larmor precession frequency
    - Q is the component index (-2, -1, 0, 1, 2)
    
    The solution is: ρ^K_Q_new = ρ^K_Q_old / (1 + i·Q·Γ)
    where Γ = ω_L / Gamma_total (dimensionless Hanle parameter)
    """
    
    # Total damping rate
    Gamma_total = Gamma_rad + Gamma_col
    
    # Dimensionless Hanle parameter
    Gamma_H = omega_L / Gamma_total  # Can be complex if Gamma_col is complex
    
    # Q values in order: [-2, -1, 0, 1, 2]
    Qs = np.array([-2, -1, 0, 1, 2])
    
    # Hanle depolarization matrix (diagonal in magnetic frame)
    H = np.zeros((5, 5), dtype=complex)
    
    for i, Q in enumerate(Qs):
        # Each diagonal element accounts for Hanle rotation + depolarization
        # The denominator (1 + i·Q·Γ_H) comes from solving the equation of motion
        H[i, i] = 1.0 / (1.0 + 1j * Q * Gamma_H)
    
    return H


def hanle_matrix_lab_frame(Gamma_rad, Gamma_col, omega_L, theta_B, chi_B):
    """
    Full Hanle matrix in LAB FRAME (irreducible tensor formalism).
    
    Transforms the diagonal Hanle matrix from magnetic frame back to lab frame
    using Wigner rotation matrices.
    
    Parameters:
    -----------
    Gamma_rad : float
        Natural decay rate (A_ul)
    Gamma_col : float
        Collisional broadening rate
    omega_L : float
        Larmor precession frequency
    theta_B : float
        Magnetic field polar angle
    chi_B : float
        Magnetic field azimuthal angle
    
    Returns:
    --------
    H_lab : 5x5 complex array
        Hanle matrix in lab frame
    
    Operation sequence:
    1. Rotate to magnetic frame: rho_mag = D^† · rho_lab
    2. Apply Hanle in magnetic frame: rho_mag_new = H_mag · rho_mag
    3. Rotate back to lab frame: rho_lab_new = D · rho_mag_new · D^†
    """
    
    # Get diagonal Hanle matrix in magnetic frame
    H_mag = hanle_matrix_magnetic_frame(Gamma_rad, Gamma_col, omega_L)
    
    # Wigner d-matrix for K=2
    d2 = wigner_d2(theta_B)
    
    # Rotation matrix to magnetic frame: D(theta_B, chi_B)
    # In matrix form: D[q, q'] = d2[q, q'] * exp(-i·q'·chi_B)
    D = np.zeros((5, 5), dtype=complex)
    Qs = [-2, -1, 0, 1, 2]
    
    for i, q in enumerate(Qs):
        for j, qp in enumerate(Qs):
            D[i, j] = d2[q, qp] * np.exp(-1j * qp * chi_B)
    
    # Inverse rotation: D_inv = D^† (conjugate transpose)
    D_inv = np.conj(D.T)
    
    # Full transformation in lab frame:
    # H_lab = D^† · H_mag · D
    H_lab = D_inv @ H_mag @ D
    
    return H_lab

def wigner_d2(theta):

    c = np.cos(theta)
    s = np.sin(theta)

    d = np.zeros((5,5), dtype=complex)

    Qs = [-2,-1,0,1,2]

    # indexing helper
    def idx(q):
        return q + 2

    # Explicit formulas

    d[idx(2),idx(2)] = (1+c)**2 / 4
    d[idx(2),idx(1)] = -(1+c)*s / 2
    d[idx(2),idx(0)] = np.sqrt(6)/4 * s**2
    d[idx(2),idx(-1)] = -(1-c)*s / 2
    d[idx(2),idx(-2)] = (1-c)**2 / 4

    d[idx(1),idx(2)] = (1+c)*s / 2
    d[idx(1),idx(1)] = (2*c**2 + c -1)/2
    d[idx(1),idx(0)] = -np.sqrt(6)/2 * s*c
    d[idx(1),idx(-1)] = (2*c**2 - c -1)/2
    d[idx(1),idx(-2)] = -(1-c)*s / 2

    d[idx(0),idx(2)] = np.sqrt(6)/4 * s**2
    d[idx(0),idx(1)] = np.sqrt(6)/2 * s*c
    d[idx(0),idx(0)] = (3*c**2 -1)/2
    d[idx(0),idx(-1)] = -np.sqrt(6)/2 * s*c
    d[idx(0),idx(-2)] = np.sqrt(6)/4 * s**2

    # symmetry relations
    for q in Qs:
        for qp in Qs:
            if d[idx(q),idx(qp)] == 0:
                d[idx(q),idx(qp)] = (
                    (-1)**(q-qp)
                    * d[idx(-q),idx(-qp)]
                )

    return d


def rotate_to_magnetic_frame(J_vert, theta_B, chi_B):

    d2 = wigner_d2(theta_B)

    J_mag = {Q: 0 for Q in [-2,-1,0,1,2]}

    for Q in J_mag:

        for Qp in J_mag:

            phase = np.exp(-1j * Qp * chi_B)

            J_mag[Q] += d2[Q, Qp] * phase * J_vert[Qp]

    return J_mag

def rotate_to_vertical_frame(S_mag, theta_B, chi_B):

    d2 = wigner_d2(theta_B)

    S_vert = {Q: 0 for Q in S_mag}

    for Q in S_mag:

        for Qp in S_mag:

            phase = np.exp(+1j * Q * chi_B)

            S_vert[Q] += d2[Qp, Q] * phase * S_mag[Qp]

    return S_vert

#-------------------------------------------------------------------------
# Additional functions for handling the density matrix and Hanle effect
Qvals = np.array([-2, -1, 0, 1, 2])

def qindex(Q):
    return Q + 2

def wigner_d2_arr(theta):

    c = np.cos(theta)
    s = np.sin(theta)

    d = np.zeros((5,5), dtype=complex)

    iq = qindex

    # Row Q, column Q'

    d[iq(2),iq(2)] = (1+c)**2/4
    d[iq(2),iq(1)] = -(1+c)*s/2
    d[iq(2),iq(0)] = np.sqrt(6)/4*s**2
    d[iq(2),iq(-1)] = -(1-c)*s/2
    d[iq(2),iq(-2)] = (1-c)**2/4

    d[iq(1),iq(2)] = (1+c)*s/2
    d[iq(1),iq(1)] = (2*c**2+c-1)/2
    d[iq(1),iq(0)] = -np.sqrt(6)/2*s*c
    d[iq(1),iq(-1)] = (2*c**2-c-1)/2
    d[iq(1),iq(-2)] = -(1-c)*s/2

    d[iq(0),iq(2)] = np.sqrt(6)/4*s**2
    d[iq(0),iq(1)] = np.sqrt(6)/2*s*c
    d[iq(0),iq(0)] = (3*c**2-1)/2
    d[iq(0),iq(-1)] = -np.sqrt(6)/2*s*c
    d[iq(0),iq(-2)] = np.sqrt(6)/4*s**2

    # symmetry
    for Q in Qvals:
        for Qp in Qvals:

            i = iq(Q)
            j = iq(Qp)

            if d[i,j] == 0:

                d[i,j] = (
                    (-1.0)**(Q-Qp)
                    * d[iq(-Q),iq(-Qp)]
                )

    return d

# TENSOR
def rotate_tensor_to_magnetic_frame(S2, theta_B, chi_B):

    d2 = wigner_d2(theta_B)

    S2_mag = np.zeros_like(S2)

    for iQ, Q in enumerate(Qvals):

        for iQp, Qp in enumerate(Qvals):

            phase = np.exp(-1j * Qp * chi_B)

            S2_mag[iQ] += (
                d2[iQ, iQp]
                * phase
                * S2[iQp]
            )

    return S2_mag

def rotate_to_magnetic_frame_arr(J2, theta_B, chi_B):

    d2 = wigner_d2_arr(theta_B)

    Jmag = np.zeros(5, dtype=complex)

    for Q in Qvals:

        iQ = qindex(Q)

        for Qp in Qvals:

            iQp = qindex(Qp)

            phase = np.exp(-1j * Qp * chi_B)

            Jmag[iQ] += (
                d2[iQ, iQp]
                * phase
                * J2[iQp]
            )

    return Jmag

# TENSOR
def rotate_tensor_to_vertical_frame(S2_mag, theta_B, chi_B):

    d2 = wigner_d2(theta_B)

    S2_vert = np.zeros_like(S2_mag)

    for iQ, Q in enumerate(Qvals):

        for iQp, Qp in enumerate(Qvals):

            phase = np.exp(+1j * Q * chi_B)

            S2_vert[iQ] += (
                d2[iQp, iQ]
                * phase
                * S2_mag[iQp]
            )

    return S2_vert

def rotate_to_vertical_frame_arr(Smag, theta_B, chi_B):

    d2 = wigner_d2_arr(theta_B)

    Svert = np.zeros(5, dtype=complex)

    for Q in Qvals:

        iQ = qindex(Q)

        for Qp in Qvals:

            iQp = qindex(Qp)

            phase = np.exp(+1j * Q * chi_B)

            Svert[iQ] += (
                d2[iQp, iQ]
                * phase
                * Smag[iQp]
            )

    return Svert

# TENSOR
def apply_hanle_effect(S2, Gamma, theta_B, chi_B):

    # rotate to magnetic frame
    S2_mag = rotate_tensor_to_magnetic_frame(
        S2,
        theta_B,
        chi_B
    )

    # Hanle operator
    for iQ, Q in enumerate(Qvals):

        S2_mag[iQ] /= (1 + 1j * Gamma * Q)

    # rotate back
    S2_vert = rotate_tensor_to_vertical_frame(
        S2_mag,
        theta_B,
        chi_B
    )

    return S2_vert

def hanle_operator_full(J2, Gamma, theta_B, chi_B):

    N_tau = J2.shape[1]

    S2 = np.zeros_like(J2)

    for t in range(N_tau):

        # -----------------------------------
        # 1. rotate to magnetic frame
        # -----------------------------------

        Jmag = rotate_to_magnetic_frame_arr(
            J2[:,t],
            theta_B,
            chi_B
        )

        # -----------------------------------
        # 2. Hanle effect in magnetic frame
        # -----------------------------------

        Smag = np.zeros(5, dtype=complex)

        for Q in Qvals:

            iQ = qindex(Q)

            Smag[iQ] = (
                Jmag[iQ]
                / (1 + 1j * Gamma * Q)
            )

        # -----------------------------------
        # 3. rotate back
        # -----------------------------------

        Svert = rotate_to_vertical_frame_arr(
            Smag,
            theta_B,
            chi_B
        )

        S2[:,t] = Svert

    return S2