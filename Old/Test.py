import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from numba import jit
from Old.functions_prt import *

# Functions

Qvals = np.array([-2, -1, 0, 1, 2])
qindex = lambda q: q + 2

def compute_J2(S2):

    J2 = np.zeros((5, N_tau), dtype=complex)

    for m, mu_m in enumerate(mu):

        w_m = w_mu[m]

        for l in range(N_chi):

            chi_l = chi[l]

            TI, _, _ = T2_emissivity_tensors(mu_m, chi_l)

            # -----------------------------------
            # Build scalar source emissivity
            # -----------------------------------

            SI = np.zeros(N_tau)

            for i in range(5):

                SI += np.real(TI[i] * S2[i])

            # add isotropic term
            SI += B

            weight = w_m * w_chi / (4*np.pi)

            for n in range(N_nu):

                tau_eff = tau * phi_nu[n]

                I_sc, _ = short_characteristics(
                    tau_eff,
                    SI,
                    mu_m,
                    B if mu_m > 0 else 0.0
                )

                # -----------------------------------
                # Project intensity onto tensors
                # -----------------------------------

                for i in range(5):

                    J2[i] += (
                        weight
                        * phi_nu[n]
                        * TI[i]
                        * I_sc
                    )

    return J2

def hanle_operator(J2, Gamma):

    J2_new = np.zeros_like(J2)

    for i, Q in enumerate(Qvals):

        J2_new[i] = J2[i] / (1 + 1j * Gamma * Q)

    return J2_new

def update_S2(J2):

    S2_new = np.zeros_like(J2)

    S2_new[qindex(0)] = (1 - epsilon) * W2 * J2[qindex(0)]

    # Q≠0 only from Hanle
    for i in range(5):

        if i != qindex(0):
            S2_new[i] = (1 - epsilon) * W2 * J2[i]

    return S2_new

def emergent(mu_m, S2):

    I, Q, U = 0.0, 0.0, 0.0

    for n in range(N_nu):

        tau_eff = tau * phi_nu[n]

        TI, TQ, TU = T2_emissivity_tensors(mu_m, 0.0)

        SI = np.real(S2[qindex(0)])

        SQ = np.zeros(N_tau)
        SU = np.zeros(N_tau)

        for i in range(5):

            SQ += np.real(TQ[i] * S2[i])
            SU += np.real(TU[i] * S2[i])

        I_sc, _ = short_characteristics(tau_eff, SI, mu_m, B if mu_m > 0 else 0.0)
        Q_sc, _ = short_characteristics(tau_eff, SQ, mu_m, 0.0)
        U_sc, _ = short_characteristics(tau_eff, SU, mu_m, 0.0)

        I += np.trapezoid(phi_nu * I_sc, x_nu)
        Q += np.trapezoid(phi_nu * Q_sc, x_nu)
        U += np.trapezoid(phi_nu * U_sc, x_nu)

    return I, Q, U

# =============================================================================
# GRID SETUP
# =============================================================================

N_tau = 91
tau = np.logspace(-4, 4, N_tau)

N_mu = 16
mu, w_mu = np.polynomial.legendre.leggauss(N_mu)

N_chi = 64  # Azimuthal points
chi = np.linspace(0, 2*np.pi, N_chi, endpoint=False)
w_chi = 2*np.pi / N_chi

N_nu = 121
x_nu = np.linspace(-5, 5, N_nu)

phi_nu = doppler_profile(x_nu)
phi_nu /= np.trapezoid(phi_nu, x_nu)

# =============================================================================
# PHYSICAL PARAMETERS
# =============================================================================

epsilon = 1e-4
B = 1.0
H2 = 1.0
W2 = 1.0
# Magnetic field parameters
B_field = 0  # Gauss
g_u = 1.0       # Landé factor

# Atomic rates
A_ul = 1e8      # s^-1 (typical strong line)
Gamma_I = 0.0   # collisions (start with 0)

# Larmor frequency
omega_L = 1.3996e6 * g_u * B_field  # s^-1

# Total damping
Gamma_total = A_ul + Gamma_I

# Hanle parameter
Gamma = omega_L / (Gamma_total + 1e-12)

S00 = np.ones(N_tau) * B
S20 = np.zeros(N_tau)

J2 = np.zeros((5, N_tau), dtype=complex)
S2 = np.zeros((5, N_tau), dtype=complex)

# only alignment
S2[qindex(0), :] = S20.copy()

theta_B = np.radians(60.0)  # magnetic field inclination
chi_B = np.radians(30.0)  # magnetic field azimuth

n_iter = 500
tol = 1e-6

for it in range(n_iter):

    S2_old = S2.copy()

    # compute radiation tensors
    J2 = compute_J2(S2)

    # full Hanle coupling
    S2 = hanle_operator_full(
        J2,
        Gamma,
        theta_B,
        chi_B
    )

    err = np.max(np.abs(S2 - S2_old))

    print(f"Iteration {it+1}, error: {err:.2e}")

    if err < tol:
        break