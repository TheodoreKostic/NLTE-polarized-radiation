import numpy as np
import matplotlib.pyplot as plt
from numba import jit
from functions_prt import *

# =============================================================================
# 1D Resonance Line Polarization with Irreducible Tensor Density Matrix
# =============================================================================
# 
# Tensor components (following Trujillo Bueno & Manso Sainz 1999):
# ρ^0_K are the irreducible tensor components of the atomic density matrix
# 
# For resonance line polarization (two-level atom, Q=0, K∈{0,2}):
#   ρ^0_0 = overall population of upper level (monopole/scalar)
#   ρ^0_2 = atomic alignment (quadrupole) - measures population imbalance among sublevels
#
# These couple to radiation field tensor moments J^0_K via statistical equilibrium:
#   J^0_0 = (1/2)∫dx φ(x)∫dμ I(x,μ)         [isotropic intensity]
#   J^0_2 = (1/(4√2))∫dx φ(x)∫dμ [(3μ²-1)I + 3(μ²-1)Q]  [anisotropic intensity]
#
# where x = frequency (Doppler units), μ = cos(θ), φ(x) = Doppler profile
# =============================================================================

# Optical depth grid
N_tau = 91
tau = np.logspace(-4, 4, N_tau)

# Angular grid (Gauss-Legendre quadrature)
N_mu = 16
mu, w_mu = np.polynomial.legendre.leggauss(N_mu)

# Frequency grid (Doppler line profile)
N_nu = 121
x_nu = np.linspace(-5, 5, N_nu)  # Frequency in Doppler widths
phi_nu = doppler_profile(x_nu)   # Gaussian profile φ(x) = exp(-x²)/√π
phi_nu /= np.trapezoid(phi_nu, x_nu)  # Normalize: ∫φ(x)dx = 1

# Parameters
epsilon = 1e-4  # thermalization parameter (collisional destruction probability)
B = 1.0         # Planck function (isothermal, B = constant)
H2 = 1.0         # depolarizing collision rate (relative to radiative decay)

# Initialize tensor source functions (frequency-integrated)
S = {
    (0,0): np.ones(N_tau) * B,        # ρ^0_0: monopole (population)
    (2,0): np.zeros(N_tau)             # ρ^0_2: quadrupole (alignment)
}

# =============================================================================
# Main iteration loop
# =============================================================================

n_iter = 50 # for ~ 50, there is a dip, for ~ 100 it already dissapears
for it in range(n_iter):

    S_old = {k: v.copy() for k, v in S.items()}

    # Initialize moment accumulators (frequency-integrated)
    J00 = np.zeros(N_tau)
    J20 = np.zeros(N_tau)
    J00_nu = np.zeros((N_tau, N_nu))  # Shape: (depth, frequency)
    J20_nu = np.zeros((N_tau, N_nu))

    I_sol = np.zeros((N_tau, N_nu))  # Store full I solution for emergent
    Q_sol = np.zeros((N_tau, N_nu))  # Store full Q solution for emergent

    # =========================================================================
    # Per-angle loop: integrate formal solution over all frequencies
    # =========================================================================
    for m in range(N_mu):

        mu_m = mu[m]
        w_m = w_mu[m]

        # Angular basis function (Legendre polynomial component)
        P2 = (3*mu_m**2 - 1)

        # =====================================================================
        # Tensor source decomposition (Eq. 8-9 from paper):
        #   S_I(τ,μ,x) = S^0_0 + (1/√2)(3μ²-1)S^0_2    [I source function]
        #   S_Q(τ,μ,x) = (3/(2√2))(1-μ²)S^0_2           [Q source function]
        # =====================================================================
        S_I_mu = S[(0,0)] + (1/np.sqrt(2)) * P2 * S[(2,0)]
        S_Q_mu = (3/(2*np.sqrt(2))) * (1 - mu_m**2) * S[(2,0)]

        # Boundary conditions for transfer equations
        I_boundary = B if mu_m > 0 else 0.0     # Thermal at depth, zero at top
        Q_boundary = 0.0                        # No polarization at boundaries

        # =====================================================================
        # Per-frequency formal solution (complete frequency redistribution)
        # =====================================================================
        # Accumulate moments across all frequencies for this angle
        J00_mu = np.zeros(N_tau)
        J20_mu = np.zeros(N_tau)
        
        dx_nu = x_nu[1] - x_nu[0]  # Frequency grid spacing
        I_store = np.zeros((N_tau, N_nu))
        Q_store = np.zeros((N_tau, N_nu))
        for n in range(N_nu):
            # In complete frequency redistribution, optical depth is frequency-independent
            # (resonance line scattering redistributes frequencies)
            tau_eff = tau * phi_nu[n]   # Effective optical depth at this frequency
            
            # Solve transfer equations: dI/dτ = S_I - I, dQ/dτ = S_Q - Q
            I_sc, _ = short_characteristics(tau_eff, S_I_mu, mu_m, I_boundary, ali=False)
            Q_sc, _ = short_characteristics(tau_eff, S_Q_mu, mu_m, Q_boundary, ali=False)

            # Store full spectra
            I_store[:, n] = I_sc
            Q_store[:, n] = Q_sc

            # Store solutions (only for m=0 upward ray for emergent evaluation)
            if m == 0 and mu_m > 0:
                I_sol[:, n] += phi_nu[n] * I_sc
                Q_sol[:, n] += phi_nu[n] * Q_sc
            
            # ================================================================
            # Accumulate moment contributions weighted by Doppler profile
            # and frequency step (trapezoidal integration)
            # ================================================================
            J00_mu = np.trapezoid(phi_nu * I_store, x_nu, axis=1)
            J20_mu = np.trapezoid(
                phi_nu * (P2 * I_store + 3*(mu_m**2 - 1) * Q_store),
                x_nu,
                axis=1)

            J00_nu[:, n] += 0.5 * w_m * phi_nu[n] * dx_nu * I_sc
            J20_nu[:, n] += (1/(4*np.sqrt(2))) * w_m * phi_nu[n] * dx_nu * (P2 * I_sc + 3*(mu_m**2 - 1) * Q_sc)
        # =====================================================================
        # Add this angle's contribution to total moments (Eq. 10-11)
        # =====================================================================
        # Angular quadrature weight: 0.5*w_μ comes from ∫_{-1}^{1} dμ = 2
        # J^0_0 = (1/2) ∫dx φ(x) ∫dμ w_μ I(τ,μ,x)
        # J^0_2 = (1/(4√2)) ∫dx φ(x) ∫dμ w_μ [(3μ²-1)I + 3(μ²-1)Q]
        
        J00 += 0.5 * w_m * J00_mu
        J20 += (1/(4*np.sqrt(2))) * w_m * J20_mu


    # =========================================================================
    # Statistical equilibrium update (Eq. 1-2 from paper)
    # =========================================================================
    # S^0_0 = (1-ε)J^0_0 + ε B  [source = non-LTE intensity + LTE correction]
    # S^0_2 = (1-ε)J^0_2        [alignment decays to zero at depth]
    S[(0,0)] = (1 - epsilon) * J00 + epsilon * B
    S[(2,0)] = (1 - epsilon) * J20 / ( 1 + H2 )

    # Bottom boundary condition: enforce isotropy deep in atmosphere
    # ρ^0_2 → 0 as τ → ∞ (populations equalize at high density)
    S[(2,0)][-1] = 0.0

    # =========================================================================
    # Convergence check
    # =========================================================================
    diff = max(
        np.max(np.abs(S[(0,0)] - S_old[(0,0)])),
        np.max(np.abs(S[(2,0)] - S_old[(2,0)]))
    )

    print(f"Iter {it}: error = {diff:.3e}")

    if diff < 1e-6:
        print(f"Converged in {it} iterations")
        break

# anisotropy (frequency-integrated)
anisotropy_depth = J20 / (J00 + 1e-12)

# Compute emergent Stokes parameters at SURFACE (τ ≈ 0, index [0])
I_emerge = np.zeros(N_mu)
Q_emerge = np.zeros(N_mu)

for m in range(N_mu):
    mu_m = mu[m]
    # Tensor basis functions MATCHING SOURCE DECOMPOSITION (Eq. 8-9)
    T_I = (1/np.sqrt(2)) * (3*mu_m**2 - 1)        # Matches line 88
    T_Q = (3/(2*np.sqrt(2))) * (1 - mu_m**2)      # Matches line 89
    
    # Emergent intensity and Q from source tensor components AT SURFACE
    I_emerge[m] = S[(0,0)][0] + T_I * S[(2,0)][0]    # Use [0] not [-1]
    Q_emerge[m] = T_Q * S[(2,0)][0]                   # Use [0] not [-1]

# Calculate Q/I ratio
Q_over_I = Q_emerge / (I_emerge + 1e-12)

# Print results
print("--------------------------------------")
print(f"Surface anisotropy J20/J00 = {anisotropy_depth[0]:.6f} ({anisotropy_depth[0]*100:.3f}%)")
print(f"Surface J20 = {J20[0]:.6e}")
print(f"Surface J00 = {J00[0]:.6e}")
print("--------------------------------------")
print("EMERGENT Q VALUES (at surface):")
print(f"Max |Q| = {np.max(np.abs(Q_emerge)):.6e}")
print(f"Min |Q| = {np.min(np.abs(Q_emerge)):.6e}")
print(f"Max |Q/I| = {np.max(np.abs(Q_over_I))*100:.3f}%")
print("--------------------------------------")
print(f"Max S20 (source): {np.max(np.abs(S[(2,0)])):.6e}")
print(f"H2 factor (Hanle depolarization): {H2:.4f}")
print("(H2=1.0 means NO magnetic field - isotropic)")
print("--------------------------------------")
print("Max S20:", np.max(np.abs(S[(2,0)])))
print("Max J20:", np.max(np.abs(J20)))
print("Max J00:", np.max(np.abs(J00)))
print('--------------------------------------')

plt.figure(figsize=(6,5))
plt.semilogx(tau, anisotropy_depth, '-o')
plt.xlabel("Optical depth τ")
plt.ylabel(r"$J^0_2 / J^0_0$")
plt.title("Anisotropy vs Depth")
plt.grid()
plt.savefig("anisotropy_tensor.png", dpi=300, bbox_inches='tight')
plt.show()

freqs_to_plot = [0, N_nu//4, N_nu//2, 3*N_nu//4, N_nu-1]  # Line core and wings

plt.figure(figsize=(10, 6))
for idx in freqs_to_plot:
    freq_label = f'x = {x_nu[idx]:.1f} Doppler widths'
    anisotropy_nu = J20_nu[:, idx] / (J00_nu[:, idx] + 1e-12)
    plt.semilogx(tau, anisotropy_nu, '-o', label=freq_label)

plt.xlabel("Optical depth τ")
plt.ylabel(r"$J^0_2 / J^0_0$ (frequency-dependent)")
plt.title("Anisotropy vs Frequency (Should show negative dip in line core)")
plt.axhline(0, color='k', linestyle='--', alpha=0.3)
plt.legend()
plt.grid()
plt.show()


# Frequency-dependent code

# =============================================================================
# 1D Resonance Line Polarization with Tensor Density Matrix (CRD)
# Frequency-resolved radiation field → frequency-resolved source function
# =============================================================================

# Optical depth grid
N_tau = 91
tau = np.logspace(-4, 4, N_tau)

# Angular grid
N_mu = 16
mu, w_mu = np.polynomial.legendre.leggauss(N_mu)

# Frequency grid
N_nu = 121
x_nu = np.linspace(-5, 5, N_nu)
phi_nu = doppler_profile(x_nu)
phi_nu /= np.trapezoid(phi_nu, x_nu)

# Parameters
epsilon = 1e-4
B = 1.0
H2 = 1.0

# Initial source functions
S = {
    (0,0): np.ones(N_tau) * B,
    (2,0): np.zeros(N_tau)
}

# =============================================================================
# ITERATION LOOP
# =============================================================================

n_iter = 50

for it in range(n_iter):

    S_old = {k: v.copy() for k, v in S.items()}

    # -------------------------------------------------------------
    # Frequency-resolved radiation field
    # -------------------------------------------------------------
    J00_nu = np.zeros((N_tau, N_nu))
    J20_nu = np.zeros((N_tau, N_nu))

    # -------------------------------------------------------------
    # Frequency-resolved source function contributions
    # -------------------------------------------------------------
    S0_nu = np.zeros((N_tau, N_nu))
    S2_nu = np.zeros((N_tau, N_nu))

    # =============================================================
    # ANGULAR LOOP
    # =============================================================
    for m in range(N_mu):

        mu_m = mu[m]
        w_m = w_mu[m]

        P2 = (3 * mu_m**2 - 1)

        # Tensor source decomposition
        S_I_mu = S[(0,0)] + (1/np.sqrt(2)) * P2 * S[(2,0)]
        S_Q_mu = (3/(2*np.sqrt(2))) * (1 - mu_m**2) * S[(2,0)]

        I_boundary = B if mu_m > 0 else 0.0
        Q_boundary = 0.0

        # =========================================================
        # FREQUENCY LOOP
        # =========================================================
        for n in range(N_nu):

            tau_eff = tau * phi_nu[n]

            I_sc, _ = short_characteristics(tau_eff, S_I_mu, mu_m, I_boundary, ali=False)
            Q_sc, _ = short_characteristics(tau_eff, S_Q_mu, mu_m, Q_boundary, ali=False)

            # -----------------------------------------------------
            # Frequency-resolved J moments
            # -----------------------------------------------------
            J00_nu[:, n] += 0.5 * w_m * I_sc

            J20_nu[:, n] += (1/(4*np.sqrt(2))) * w_m * (
                (3*mu_m**2 - 1) * I_sc +
                3*(mu_m**2 - 1) * Q_sc
            )

    # =============================================================
    # FREQUENCY-DEPENDENT SOURCE FUNCTION
    # =============================================================

    S0_nu = (1 - epsilon) * J00_nu + epsilon * B
    S2_nu = (1 - epsilon) * J20_nu / (1 + H2)

    # =============================================================
    # COLLAPSE TO PHYSICAL SOURCE FUNCTION
    # =============================================================

    S[(0,0)] = np.trapezoid(phi_nu * S0_nu, x_nu, axis=1)
    S[(2,0)] = np.trapezoid(phi_nu * S2_nu, x_nu, axis=1)

    # enforce boundary condition
    S[(2,0)][-1] = 0.0

    # =============================================================
    # CONVERGENCE
    # =============================================================

    diff = max(
        np.max(np.abs(S[(0,0)] - S_old[(0,0)])),
        np.max(np.abs(S[(2,0)] - S_old[(2,0)]))
    )

    print(f"Iter {it}: error = {diff:.3e}")

    if diff < 1e-6:
        print(f"Converged in {it} iterations")
        break

# =============================================================================
# POSTPROCESSING
# =============================================================================

anisotropy = S[(2,0)] / (S[(0,0)] + 1e-12)

# emergent intensity
I_emerge = np.zeros(N_mu)
Q_emerge = np.zeros(N_mu)

for m in range(N_mu):

    mu_m = mu[m]

    T_I = (1/np.sqrt(2)) * (3*mu_m**2 - 1)
    T_Q = (3/(2*np.sqrt(2))) * (1 - mu_m**2)

    I_emerge[m] = S[(0,0)][0] + T_I * S[(2,0)][0]
    Q_emerge[m] = T_Q * S[(2,0)][0]

Q_over_I = Q_emerge / (I_emerge + 1e-12)

# =============================================================================
# OUTPUT
# =============================================================================

print("--------------------------------------")
print(f"Surface anisotropy = {anisotropy[0]:.6e}")
print(f"Max Q/I = {np.max(np.abs(Q_over_I))*100:.3f}%")
print("--------------------------------------")

# =============================================================================
# PLOTS
# =============================================================================

plt.figure(figsize=(6,5))
plt.semilogx(tau, anisotropy, '-o')
plt.xlabel("Optical depth τ")
plt.ylabel(r"$J^0_2 / J^0_0$")
plt.title("Anisotropy (frequency-resolved source method)")
plt.grid()
plt.show()

# =============================================================================
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

# =============================================================================
# INITIAL CONDITIONS
# =============================================================================

S = {
    (0,0): np.ones(N_tau) * B,
    (2,0): np.zeros(N_tau)
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

        P2 = (3 * mu_m**2 - 1)

        # Source functions
        S_I_mu = S[(0,0)] + (1/np.sqrt(2)) * P2 * S[(2,0)]
        S_Q_mu = (3/(2*np.sqrt(2))) * (1 - mu_m**2) * S[(2,0)]

        I_boundary = B if mu_m > 0 else 0.0
        Q_boundary = 0.0

        # Storage over frequency
        I_store = np.zeros((N_tau, N_nu))
        Q_store = np.zeros((N_tau, N_nu))
        L_I_store = np.zeros((N_tau, N_nu))
        L_Q_store = np.zeros((N_tau, N_nu))

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
                tau_eff, S_I_mu, mu_m, I_boundary, ali=True
            )

            Q_sc, L_Q = short_characteristics(
                tau_eff, S_Q_mu, mu_m, Q_boundary, ali=True
            )

            I_store[:, n] = I_sc
            Q_store[:, n] = Q_sc
            L_I_store[:, n] = L_I
            L_Q_store[:, n] = L_Q

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

        # =================================================================
        # ANGULAR INTEGRATION
        # =================================================================

        J00 += 0.5 * w_m * J00_mu
        J20 += (1/(4*np.sqrt(2))) * w_m * J20_mu

        Lambda00 += 0.5 * w_m * Lambda00_mu
        Lambda20 += (1/(4*np.sqrt(2))) * w_m * Lambda20_mu

    # =====================================================================
    # ALI UPDATE
    # =====================================================================

    J00_eff = J00 - Lambda00 * S_old[(0,0)]
    J20_eff = J20 - Lambda20 * S_old[(2,0)]

    denom00 = 1.0 - (1 - epsilon) * Lambda00
    denom00 = np.where(np.abs(denom00) < eps_denom, eps_denom, denom00)

    S[(0,0)] = ((1 - epsilon) * J00_eff + epsilon * B) / denom00

    denom20 = 1.0 - (1 - epsilon) * Lambda20 / (1 + H2)
    denom20 = np.where(np.abs(denom20) < eps_denom, eps_denom, denom20)

    S[(2,0)] = ((1 - epsilon) * J20_eff / (1 + H2)) / denom20

    if force_S20_bottom_zero:
        S[(2,0)][-1] = 0.0

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
for m in range(N_mu):

    mu_m = mu[m]

    T_I = (1/np.sqrt(2)) * (3*mu_m**2 - 1)
    T_Q = (3/(2*np.sqrt(2))) * (1 - mu_m**2)

    I_emerge[m] = S[(0,0)][0] + T_I * S[(2,0)][0]
    Q_emerge[m] = T_Q * S[(2,0)][0] 
Q_over_I = Q_emerge / (I_emerge + 1e-12)
print("--------------------------------------")
print(f"Surface anisotropy = {anisotropy[0]*100:.3f}%")
print(f"Max Q/I = {np.max(np.abs(Q_over_I))*100:.3f}%")
print("--------------------------------------") 
