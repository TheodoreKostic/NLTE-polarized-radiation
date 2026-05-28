import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from numba import jit
from functions_prt import *
from copy import deepcopy

# =============================================================================
# General version of the code for the 1D case. It is used to test the density matrix solver.
# -----------------------------------------------------------------------------/

# Local functions
# ----------------
Qvals = np.array([-2, -1, 0, 1, 2])

def irreducible_weights(mu):

    T00 = 1.0

    T20 = (1/(2*np.sqrt(2))) * (3*mu**2 - 1)

    return T00, T20

def compute_radiation_tensor(S00, S20):

    J00 = np.zeros(N_tau)
    J20 = np.zeros(N_tau)

    for m in range(N_mu):


        mu_m = mu[m]
        w_m = w_mu[m]

        for l in range(N_chi):

            chi_l = chi[l]

            weight = w_m * w_chi / (4*np.pi)

            S_I = scalar_source(S, mu_m, chi_l)

            I_store = np.zeros((N_tau, N_nu))

        for n in range(N_nu):

                tau_eff = tau * phi_nu[n]
                I_boundary = B if mu_m > 0 else 0.0
                I_sc, _ = short_characteristics(
                    tau_eff,
                    S_I,
                    mu_m,
                    I_boundary
                )

                I_store[:,n] = I_sc

            Ibar = np.trapezoid(
                phi_nu[None,:] * I_store,
                x_nu,
                axis=1
            )

            J00 += weight * T00(mu_m, chi_l) * Ibar

            J20 += weight * T20(mu_m, chi_l) * Ibar

            J22 += weight * np.conj(
                T22(mu_m, chi_l)
            ) * Ibar

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

# ============================================================
# GRID
# ============================================================

N_tau = 91
tau = np.logspace(-4, 4, N_tau)

N_mu = 16
mu_full, w_full = np.polynomial.legendre.leggauss(32)

mu = mu_full
w_mu = w_full
N_mu = len(mu)

#mu = mu_full[mu_full > 0]
#w_mu = w_full[mu_full > 0]

N_chi = 32
chi = np.linspace(0, 2*np.pi, N_chi, endpoint=False)
w_chi = 2*np.pi / N_chi

N_nu = 121
x_nu = np.linspace(-5, 5, N_nu)

phi_nu = doppler_profile(x_nu)
phi_nu /= np.trapezoid(phi_nu, x_nu)

# ============================================================
# PHYSICAL PARAMETERS
# ============================================================

epsilon = 1e-4
B = 1.0
W2 = 1.0

theta_B = np.radians(60.0)
chi_B = np.radians(30.0)

# Hanle parameters
B_field = 10.0     # Gauss
g_u = 1.0
A_ul = 1e8
Gamma_I = 0.0

omega_L = 1.3996e6 * g_u * B_field
Gamma = omega_L / (A_ul + Gamma_I)

# ============================================================
# LANDI TENSORS
# ============================================================

def T00(mu, chi):
    return 1.0

def T20(mu, chi):
    return (1.0/(2*np.sqrt(2))) * (3*mu**2 - 1)

def T22(mu, chi):
    return np.sqrt(3/8) * (1 - mu**2) * np.exp(2j*chi)

def T2m2(mu, chi):
    return np.conj(T22(mu, chi))

# ============================================================
# BUILD SOURCE FUNCTION
# ============================================================

def scalar_source(S, mu, chi):

    return (
        T00(mu, chi) * S[(0,0)]
        + T20(mu, chi) * S[(2,0)]
        + 2*np.real(
            T22(mu, chi) * S[(2,2)]
        )
    ).real

# ============================================================
# SOURCE FUNCTION UPDATE
# ============================================================

def update_source_function(J00, J20, J22):

    S00 = (1-epsilon) * J00 + epsilon * B

    S20 = (1-epsilon) * W2 * J20

    # Hanle depolarization + rotation
    S22 = (
        (1-epsilon) * W2 * J22
        / (1 + 2j*Gamma)
    )

    return {
        (0,0): S00,
        (2,0): S20,
        (2,2): S22
    }

# ============================================================
# EMERGENT STOKES
# ============================================================
def emergent_stokes(S, mu_obs, chi_obs):

    S00 = S[(0,0)][0]
    S20 = S[(2,0)][0]
    S22 = S[(2,2)][0]

    I = (
        S00
        + T20(mu_obs, chi_obs)*S20
        + 2*np.real(
            T22(mu_obs, chi_obs)*S22
        )
    )

    Q = (
        (3/(2*np.sqrt(2)))
        * (1-mu_obs**2)
        * S20

        - np.sqrt(3)
        * (1+mu_obs**2)
        * np.real(
            np.exp(2j*chi_obs)*S22
        )
    )

    U = (
        np.sqrt(3)
        * mu_obs
        * np.imag(
            np.exp(2j*chi_obs)*S22
        )
    )

    return I, Q, U

# ============================================================
# INITIAL CONDITIONS
# ============================================================

S = {
    (0,0): np.ones(N_tau)*B,

    (2,-2): np.zeros(N_tau, dtype=complex),
    (2,-1): np.zeros(N_tau, dtype=complex),
    (2,0):  np.zeros(N_tau, dtype=complex),
    (2,1):  np.zeros(N_tau, dtype=complex),
    (2,2):  np.zeros(N_tau, dtype=complex),
}

# ============================================================
# PRECOMPUTE HANLE MATRIX
# ============================================================

# Compute Hanle matrices for use in iteration
# Note: We need both magnetic frame (for direct application) and lab frame versions
H_mag = hanle_matrix_magnetic_frame(
    Gamma_rad=A_ul,
    Gamma_col=0.0,  # Set to non-zero if collisional broadening is included
    omega_L=omega_L
)

H_lab = hanle_matrix_lab_frame(
    Gamma_rad=A_ul,
    Gamma_col=0.0,
    omega_L=omega_L,
    theta_B=theta_B,
    chi_B=chi_B
)

print(f"Hanle parameters:")
print(f"  ω_L = {omega_L:.3e} rad/s")
print(f"  Γ = {Gamma:.3e}")
print(f"  Dimensionless Hanle parameter: Γ_H = ω_L / A_ul = {omega_L/A_ul:.3e}")
print(f"\nHanle matrix (magnetic frame, diagonal):")
print(f"  H_mag[0,0] (Q=-2): {H_mag[0,0]:.4f}")
print(f"  H_mag[2,2] (Q= 0): {H_mag[2,2]:.4f}")
print(f"  H_mag[4,4] (Q=+2): {H_mag[4,4]:.4f}")

# ============================================================
# SCALAR NLTE ITERATION
# ============================================================

n_iter = 100
tol = 1e-6

for it in range(n_iter):

    S_old = deepcopy(S)

    # -----------------------------------------
    # formal solution + radiation tensors
    # -----------------------------------------
    J00, J20, J22 = compute_radiation_tensor(S)

    # -----------------------------------------
    # build irreducible vector (Q=-2,-1,0,1,2)
    # -----------------------------------------
    J_vert = {
        -2: np.conj(J22),
        -1: np.zeros_like(J00),
         0: J20,
         1: np.zeros_like(J00),
         2: J22
    }

    # -----------------------------------------
    # Apply full Hanle matrix in lab frame
    # -----------------------------------------
    # Stack J components: J_arr[i, tau_idx] = J_vert[Q_i][tau_idx]
    J_arr = np.array([J_vert[q] for q in [-2,-1,0,1,2]])  # shape (5, N_tau)
    
    # Apply Hanle matrix for each depth point:
    # S_arr[i, tau_idx] = (1-epsilon)*W2 * sum_j H_lab[i,j] * J_arr[j, tau_idx]
    S_arr = (1-epsilon)*W2 * np.dot(H_lab, J_arr)
    
    # Convert back to dict
    S_quad = {}
    for i, Q in enumerate([-2,-1,0,1,2]):
        S_quad[Q] = S_arr[i]

    # Scalar component (no Hanle effect for K=0)
    S00 = (1-epsilon)*J00 + epsilon*B

    # -----------------------------------------
    # update source tensors
    # -----------------------------------------
    S[(0,0)] = S00

    for Q in [-2,-1,0,1,2]:
        S[(2,Q)] = S_quad[Q]

    # -----------------------------------------
    # convergence
    # -----------------------------------------
    err = max(
        np.max(np.abs(S[k] - S_old[k]))
        for k in S
    )

    print(f"Iter {it+1}, err = {err:.2e}")

    if err < tol:
        print("Converged.")
        break

# ============================================================
# EMERGENT POLARIZATION
# ============================================================

Q_I = np.zeros(N_mu)
U_I = np.zeros(N_mu)

I_arr = np.zeros(N_mu)
Q_arr = np.zeros(N_mu)
U_arr = np.zeros(N_mu)

chi_obs = 0.0
for m in range(N_mu):

    mu_obs = mu[m]

    I, Q, U = emergent_stokes(
        S,
        mu_obs,
        chi_obs
    )

    I_arr[m] = I
    Q_arr[m] = Q
    U_arr[m] = U

    Q_I[m] = Q / (I + 1e-12)
    U_I[m] = U / (I + 1e-12)

# ============================================================
# PLOTS
# ============================================================

plt.figure(figsize=(6,5))
plt.plot(mu, Q_I*100, '-o', label='Q/I')
plt.plot(mu, U_I*100, '-o', label='U/I')
plt.xlabel(r'$\mu$')
plt.ylabel('Polarization (%)')
plt.legend()
plt.grid()
plt.show()

print("Q_I:", Q_I)
print("U_I:", U_I)
# Anisotropy at the surface
anisotropy = J20 / (J00 + 1e-12)
print("Anisotropy (J20/J00):", anisotropy[0])
plt.figure(figsize=(6,5))
plt.plot(tau, anisotropy, '-o')
plt.xscale('log')
plt.xlabel(r'$\tau$')
plt.ylabel('Anisotropy (J20/J00)')
plt.grid()
plt.show()

print("Max |J20|:", np.max(np.abs(J20)))
print("Max |J22|:", np.max(np.abs(J22)))

plt.figure(figsize=(6,5))
plt.plot(mu, I_arr, '-o', label='I')
plt.plot(mu, Q_arr, '-o', label='Q')
plt.plot(mu, U_arr, '-o', label='U')
plt.xlabel(r'$\mu$')
plt.legend()
plt.grid()
plt.show()


print("surface J00, J20, |J22| =", J00[0], J20[0], np.abs(J22[0]))