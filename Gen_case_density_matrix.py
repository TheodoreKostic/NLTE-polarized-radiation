import numpy as np
import matplotlib.pyplot as plt
from numba import jit
from functions_prt import *

# =============================================================================
# General version of the code for the 1D case. It is used to test the density matrix solver.
# -----------------------------------------------------------------------------/

# # =============================================================================
# 1D Resonance Line Polarization — FINAL ALI IMPLEMENTATION
# Uses TRUE diagonal Lambda* from short_characteristics_ALI()
# =============================================================================

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
B_field = 10.0  # Gauss
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
# INITIAL CONDITIONS
# =============================================================================

S = {
    (0,0): np.ones(N_tau) * B,
    (2,0): np.zeros(N_tau),
    (2,2): np.zeros(N_tau, dtype=complex)  # NEW
}

# =============================================================================
# MAIN ALI ITERATION
# =============================================================================

n_iter = 1000
tol = 1e-6
eps_denom = 1e-10
force_S20_bottom_zero = True

for it in range(n_iter):

    S_old = {k: v.copy() for k, v in S.items()}

    J00 = np.zeros(N_tau)
    J20 = np.zeros(N_tau)

    Lambda00 = np.zeros(N_tau)
    Lambda20 = np.zeros(N_tau)

    # =====================================================================
    # ANGLE LOOP
    # =====================================================================
    for m in range(N_mu):

        mu_m = mu[m]
        w_m = w_mu[m]
        for l in range(N_chi):
            chi_l = chi[l]

            P2 = 0.5 * (3 * mu_m**2 - 1)

            cos2chi = np.cos(2 * chi_l)
            sin2chi = np.sin(2 * chi_l)

            S00 = S[(0,0)]
            S20 = S[(2,0)]
            S22 = S[(2,2)]

            ReS22 = np.real(S22)
            ImS22 = np.imag(S22)

            S_I = (
                S00
                + (1/np.sqrt(2)) * P2 * S20
                - np.sqrt(3) * (1 - mu_m**2) * (cos2chi * ReS22 - sin2chi * ImS22)
            )

            S_Q = (
                (3/(2*np.sqrt(2))) * (1 - mu_m**2) * S20
                - np.sqrt(3) * (1 + mu_m**2) * (cos2chi * ReS22 - sin2chi * ImS22)
            )

            S_U = (
                np.sqrt(3) * mu_m * (sin2chi * ReS22 + cos2chi * ImS22)
            )

        I_boundary = B if mu_m > 0 else 0.0
        Q_boundary = 0.0

        # Storage over frequency
        I_store = np.zeros((N_tau, N_nu))
        Q_store = np.zeros((N_tau, N_nu))
        U_store = np.zeros((N_tau, N_nu))

        L_I_store = np.zeros((N_tau, N_nu))
        L_Q_store = np.zeros((N_tau, N_nu))
        L_U_store = np.zeros((N_tau, N_nu))

        # Chain rule factors (fixed per μ)
        a = (1/np.sqrt(2)) * P2
        b = (3/(2*np.sqrt(2))) * (1 - mu_m**2)

        # =================================================================
        # FREQUENCY LOOP
        # =================================================================
        for n in range(N_nu):

            phi = phi_nu[n]
            tau_eff = tau * phi

            I_sc, L_I = short_characteristics(
                tau_eff, S_I, mu_m, I_boundary, ali=True
            )
            
            Q_sc, L_Q = short_characteristics(
                tau_eff, S_Q, mu_m, Q_boundary, ali=True
            )
            U_sc, L_U = short_characteristics(tau_eff, S_U, mu_m, 0.0, ali=True)

            I_store[:, n] = I_sc
            Q_store[:, n] = Q_sc
            U_store[:, n] = U_sc
            L_I_store[:, n] = L_I
            L_Q_store[:, n] = L_Q
            L_U_store[:, n] = L_U

        # =================================================================
        # FREQUENCY INTEGRATION (ONLY ONCE — CORRECT)
        # =================================================================

        J00_mu = np.trapezoid(
            phi_nu[None, :] * I_store,
            x_nu,
            axis=1
        )

        J20_mu = np.trapezoid(
            phi_nu[None, :] * (
                P2 * I_store + 3*(mu_m**2 - 1) * Q_store
            ),
            x_nu,
            axis=1
        )
        J22_real_mu = np.trapezoid(
            phi_nu[None, :] * (
                ( (1 - mu_m**2) * cos2chi ) * I_store
                - ( (1 + mu_m**2) * cos2chi ) * Q_store
                - (2 * mu_m * sin2chi) * U_store
            ),
            x_nu,
            axis=1
        )

        J22_imag_mu = np.trapezoid(
            phi_nu[None, :] * (
                ( (1 - mu_m**2) * sin2chi ) * I_store
                - ( (1 + mu_m**2) * sin2chi ) * Q_store
                + (2 * mu_m * cos2chi) * U_store
            ),
            x_nu,
            axis=1
        )
        Lambda00_mu = np.trapezoid(
            phi_nu[None, :] * L_I_store,
            x_nu,
            axis=1
        )

        Lambda20_mu = np.trapezoid(
            phi_nu[None, :] * (
                P2 * (a * L_I_store) +
                3*(mu_m**2 - 1) * (b * L_Q_store)
            ),
            x_nu,
            axis=1
        )
        J22_real = np.zeros(N_tau)
        J22_imag = np.zeros(N_tau)
        # =================================================================
        # ANGULAR INTEGRATION
        # =================================================================
        weight = w_mu[m] * w_chi / (4*np.pi)

        J00 += weight * J00_mu
        J20 += weight * J20_mu

        J22_real += weight * J22_real_mu
        J22_imag += weight * J22_imag_mu

        Lambda00 += 0.5 * w_m * Lambda00_mu
        Lambda20 += (1/(4*np.sqrt(2))) * w_m * Lambda20_mu

        Gamma = (1.3996e6 * g_u * B_field) / (A_ul + Gamma_I)

        H = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, Gamma],
            [0.0, -Gamma, 1.0]
        ]) / (1.0 + Gamma**2)

    # =====================================================================
    # ALI UPDATE
    # =====================================================================

    S20_new = np.zeros(N_tau)
    S22_new = np.zeros(N_tau, dtype=complex)

    for i in range(N_tau):

        J_vec = np.array([
            J20[i],
            J22_real[i],
            J22_imag[i]
        ])

        S_vec = (1 - epsilon) * (H @ J_vec)

        S20_new[i] = S_vec[0]
        S22_new[i] = S_vec[1] + 1j * S_vec[2]

    S[(2,0)] = S20_new
    S[(2,2)] = S22_new

    if force_S20_bottom_zero:
        S[(2,0)][-1] = 0.0
        S[(2,2)][-1] = 0.0 + 0.0j

    # =====================================================================
    # CONVERGENCE
    # =====================================================================

    rel00 = np.max(
        np.abs((S[(0,0)] - S_old[(0,0)]) / (S_old[(0,0)] + 1e-12))
    )

    rel20 = np.max(
        np.abs(S[(2,0)] - S_old[(2,0)])
    )

    diff = max(rel00, rel20)

    print(f"Iter {it}: err = {diff:.3e}")

    if diff < tol:
        print(f"Converged in {it} iterations")
        break

# =============================================================================
# DIAGNOSTICS
# =============================================================================

anisotropy = J20 / (J00 + 1e-12)

plt.figure(figsize=(6,5))
plt.semilogx(tau, anisotropy, '-o')
plt.xlabel("Optical depth τ")
plt.ylabel(r"$J^0_2 / J^0_0$")
plt.title("Anisotropy vs Depth (ALI)")
plt.grid()
plt.show()

# Emergent Stokes parameters
I_emerge = np.zeros(N_mu)
Q_emerge = np.zeros(N_mu)
U_emerge = np.zeros(N_mu)

for m in range(N_mu):

    mu_m = mu[m]
    chi_l = 0.0

    P2 = 0.5 * (3 * mu_m**2 - 1)
    cos2chi = np.cos(2 * chi_l)
    sin2chi = np.sin(2 * chi_l)

    S00 = S[(0,0)]
    S20 = S[(2,0)]
    S22 = S[(2,2)]

    ReS22 = np.real(S22)
    ImS22 = np.imag(S22)

    S_I = (
        S00
        + (1/np.sqrt(2)) * P2 * S20
        - np.sqrt(3) * (1 - mu_m**2) * (cos2chi * ReS22 - sin2chi * ImS22)
    )

    S_Q = (
        (3/(2*np.sqrt(2))) * (1 - mu_m**2) * S20
        - np.sqrt(3) * (1 + mu_m**2) * (cos2chi * ReS22 - sin2chi * ImS22)
    )

    S_U = (
        np.sqrt(3) * mu_m *
        (sin2chi * ReS22 + cos2chi * ImS22)
    )

    I_nu = np.zeros(N_nu)
    Q_nu = np.zeros(N_nu)
    U_nu = np.zeros(N_nu)

    for n in range(N_nu):

        phi = phi_nu[n]
        tau_eff = tau * phi

        I_sc, _ = short_characteristics(tau_eff, S_I, mu_m, B)
        Q_sc, _ = short_characteristics(tau_eff, S_Q, mu_m, 0.0)
        U_sc, _ = short_characteristics(tau_eff, S_U, mu_m, 0.0)

        I_nu[n] = I_sc[0]   # emergent = surface point
        Q_nu[n] = Q_sc[0]
        U_nu[n] = U_sc[0]

    I_emerge[m] = np.trapezoid(phi_nu * I_nu, x_nu)
    Q_emerge[m] = np.trapezoid(phi_nu * Q_nu, x_nu)
    U_emerge[m] = np.trapezoid(phi_nu * U_nu, x_nu)

Q_over_I = Q_emerge / (I_emerge + 1e-12)
U_over_I = U_emerge / (I_emerge + 1e-12)

flux = np.sum(mu * I_emerge * w_mu)

print("------------------------------")
mask = mu > 0
flux = np.sum(mu[mask] * I_emerge[mask] * w_mu[mask])
print(f"Emergent flux = {flux:.3f}")
print("------------------------------")

print("--------------------------------------")
print(f"Surface anisotropy = {anisotropy[0]*100:.3f}%")
print(f"Emergent I = {I_emerge}")
print("------------------------------")
print(f"Q = {Q_emerge}")
print(f"Max Q/I = {np.max(np.abs(Q_over_I))*100:.3f}%")
print("------------------------------")
print(f"U = {U_emerge}")
print(f"Max U/I = {np.max(np.abs(U_over_I))*100:.3f}%")
print("--------------------------------------") 

plt.figure(figsize=(6,5))
plt.plot(mu, Q_over_I * 100, '-o', label='Q/I (%)')
plt.plot(mu, U_over_I * 100, '-o', label='U/I (%)')
plt.xlabel(r"Cosine of LOS angle μ")
plt.ylabel("Fractional Polarization (%)")
plt.title("Emergent Polarization vs LOS Angle")
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(6,5))
plt.semilogx(tau, S[(2,0)])
plt.semilogx(tau, np.abs(S[(2,2)]))
plt.xlabel("Optical depth τ")
plt.ylabel("Source Function Components")
plt.title("Source Function Components vs Depth")
plt.legend([r"$S^2_0$", r"$|S^2_2|$"])
plt.grid()
plt.show()  