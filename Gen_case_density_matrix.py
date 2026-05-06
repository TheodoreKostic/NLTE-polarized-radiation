import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from numba import jit
from functions_prt import *

# =============================================================================
# General version of the code for the 1D case. It is used to test the density matrix solver.
# -----------------------------------------------------------------------------/

# # =============================================================================
# 1D Resonance Line Polarization — FINAL ALI IMPLEMENTATION
# Uses TRUE diagonal Lambda* from short_characteristics_ALI()
# =============================================================================

# Local functions
def irreducible_weights(mu, chi):
    """
    Returns geometric tensors T^K_Q for K=0,2 in 1D case.
    """

    T00 = 1.0

    T20 = 0.5 * (3*mu**2 - 1)

    T22_re = (1 - mu**2) * np.cos(2*chi)
    T22_im = (1 - mu**2) * np.sin(2*chi)

    return T00, T20, T22_re, T22_im

def compute_radiation_tensor(S):

    J00 = np.zeros(N_tau)
    J20 = np.zeros(N_tau)
    J22 = np.zeros(N_tau, dtype=complex)

    for m in range(N_mu):
        mu_m = mu[m]
        w_m = w_mu[m]

        for l in range(N_chi):
            chi_l = chi[l]

            T00, T20, T22r, T22i = irreducible_weights(mu_m, chi_l)

            weight_ang = w_m * w_chi / (4*np.pi)

            I_store = np.zeros((N_tau, N_nu))

            # Only I is needed in irreducible formulation
            S_I = build_scalar_source(S, mu_m, chi_l)

            for n in range(N_nu):

                phi = phi_nu[n]
                tau_eff = tau * phi

                I_sc, _ = short_characteristics(
                    tau_eff, S_I, mu_m, B
                )

                I_store[:, n] = I_sc

            J00 += weight_ang * np.trapezoid(
                phi_nu[None,:] * T00 * I_store,
                x_nu, axis=1
            )

            J20 += weight_ang * np.trapezoid(
                phi_nu[None,:] * T20 * I_store,
                x_nu, axis=1
            )

            J22 += weight_ang * np.trapezoid(
                phi_nu[None,:] * (T22r + 1j*T22i) * I_store,
                x_nu, axis=1
            )

    return J00, J20, J22

def build_scalar_source(S, mu, chi):

    S00 = S[(0,0)]
    S20 = S[(2,0)]
    S22 = S[(2,2)]

    Re = np.real(S22)
    Im = np.imag(S22)

    T20 = 0.5*(3*mu**2 - 1)
    T22r = (1 - mu**2)*np.cos(2*chi)
    T22i = (1 - mu**2)*np.sin(2*chi)

    return (
        S00
        + T20 * S20
        + T22r * Re
        + T22i * Im
    )

def compute_emergent_stokes(S):

    I_emerge = np.zeros(N_mu)
    Q_emerge = np.zeros(N_mu)
    U_emerge = np.zeros(N_mu)

    for m in range(N_mu):

        mu_m = mu[m]

        P2 = 0.5 * (3 * mu_m**2 - 1)
        chi_l = 0.0  # disk plane

        cos2chi = np.cos(2 * chi_l)
        sin2chi = np.sin(2 * chi_l)

        S00 = S[(0,0)]
        S20 = S[(2,0)]
        S22 = S[(2,2)]

        ReS22 = np.real(S22)
        ImS22 = np.imag(S22)

        I_nu = np.zeros(N_nu)
        Q_nu = np.zeros(N_nu)
        U_nu = np.zeros(N_nu)

        for n in range(N_nu):

            phi = phi_nu[n]
            tau_eff = tau * phi

            S_I = (
                S00
                + (1/np.sqrt(2)) * P2 * S20
                - np.sqrt(3)*(1 - mu_m**2) *
                  (cos2chi*ReS22 - sin2chi*ImS22)
            )

            S_Q = (
                (3/(2*np.sqrt(2))) * (1 - mu_m**2) * S20
                - np.sqrt(3)*(1 + mu_m**2) *
                  (cos2chi*ReS22 - sin2chi*ImS22)
            )

            S_U = (
                np.sqrt(3)*mu_m *
                (sin2chi*ReS22 + cos2chi*ImS22)
            )

            I_sc, _ = short_characteristics(tau_eff, S_I, mu_m, B)
            Q_sc, _ = short_characteristics(tau_eff, S_Q, mu_m, 0.0)
            U_sc, _ = short_characteristics(tau_eff, S_U, mu_m, 0.0)

            I_nu[n] = I_sc[0]
            Q_nu[n] = Q_sc[0]
            U_nu[n] = U_sc[0]

        # frequency integration
        I_emerge[m] = np.trapezoid(phi_nu * I_nu, x_nu)
        Q_emerge[m] = np.trapezoid(phi_nu * Q_nu, x_nu)
        U_emerge[m] = np.trapezoid(phi_nu * U_nu, x_nu)

    return I_emerge, Q_emerge, U_emerge

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

# =============================================================================
# ============================================================
# INITIAL CONDITIONS
# ============================================================

S = {
    (0,0): np.ones(N_tau) * B,
    (2,0): np.zeros(N_tau),
    (2,2): np.zeros(N_tau, dtype=complex)
}

# Hanle matrix (constant)
Gamma = (1.3996e6 * g_u * B_field) / (A_ul + Gamma_I)


# ============================================================
# ITERATION LOOP
# ============================================================

n_iter = 200
tol = 1e-6
W2 = 1

for it in range(n_iter):

    S_old = {k: v.copy() for k,v in S.items()}

    J00, J20, J22 = compute_radiation_tensor(S)

    S00_new = (1 - epsilon) * J00 + epsilon * B
    S20_new = (1 - epsilon) * W2 * J20
    S22_new = (1 - epsilon) * W2 * J22

    S[(0,0)] = S00_new
    S[(2,0)] = S20_new
    S[(2,2)] = S22_new

    # boundary condition
    S[(2,0)][-1] = 0.0
    S[(2,2)][-1] = 0.0 + 0j

    err = max(
        np.max(np.abs(S[(0,0)] - S_old[(0,0)])),
        np.max(np.abs(S[(2,0)] - S_old[(2,0)])),
        np.max(np.abs(S[(2,2)] - S_old[(2,2)]))
    )

    print(f"Iter {it}: err = {err:.3e}")

    if err < tol:
        break

I_emerge, Q_emerge, U_emerge = compute_emergent_stokes(S)
Q_over_I = Q_emerge / (I_emerge + 1e-12)
U_over_I = U_emerge / (I_emerge + 1e-12)

plt.figure(figsize=(6,5))

plt.plot(mu, Q_over_I * 100, '-o', label='Q/I (%)')
plt.plot(mu, U_over_I * 100, '-o', label='U/I (%)')

plt.xlabel(r"Cosine of LOS angle $\mu$")
plt.ylabel("Fractional polarization (%)")
plt.title("Scattering Polarization CLV")
plt.legend()
plt.grid()

plt.show()

plt.figure(figsize=(6,5))

plt.plot(mu, I_emerge, '-o', color='black')

plt.xlabel(r"$\mu$")
plt.ylabel("Emergent Intensity")
plt.title("Center-to-limb intensity variation")
plt.grid()

plt.show()

plt.figure(figsize=(6,5))

plt.plot(mu, I_emerge, label='I')
plt.plot(mu, Q_emerge, label='Q')
plt.plot(mu, U_emerge, label='U')

plt.xlabel(r"$\mu$")
plt.title("Emergent Stokes Parameters")
plt.legend()
plt.grid()

plt.show()

line_center = N_nu // 2

Q_I_line = Q_over_I

plt.figure(figsize=(6,5))

plt.plot(mu, Q_I_line * 100, '-o')

plt.xlabel(r"$\mu$")
plt.ylabel("Q/I (%) at line center")
plt.title("Scattering polarization signal")
plt.grid()

plt.show()