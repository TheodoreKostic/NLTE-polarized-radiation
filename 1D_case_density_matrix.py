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
N_tau = 50
tau = np.logspace(-4, 8, N_tau)

# Angular grid (Gauss-Legendre quadrature)
N_mu = 12
mu, w_mu = np.polynomial.legendre.leggauss(N_mu)

# Frequency grid (Doppler line profile)
N_nu = 81
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

n_iter = 1000
for it in range(n_iter):

    S_old = {k: v.copy() for k, v in S.items()}

    # Initialize moment accumulators (frequency-integrated)
    J00 = np.zeros(N_tau)
    J20 = np.zeros(N_tau)

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
        
        for n in range(N_nu):
            # In complete frequency redistribution, optical depth is frequency-independent
            # (resonance line scattering redistributes frequencies)
            tau_eff = tau
            
            # Solve transfer equations: dI/dτ = S_I - I, dQ/dτ = S_Q - Q
            I_sc = short_characteristics(tau_eff, S_I_mu, mu_m, I_boundary)
            Q_sc = short_characteristics(tau_eff, S_Q_mu, mu_m, Q_boundary)

            # Store solutions (only for m=0 upward ray for emergent evaluation)
            if m == 0 and mu_m > 0:
                I_sol[:, n] += phi_nu[n] * I_sc
                Q_sol[:, n] += phi_nu[n] * Q_sc
            
            # ================================================================
            # Accumulate moment contributions weighted by Doppler profile
            # and frequency step (trapezoidal integration)
            # ================================================================
            J00_mu += phi_nu[n] * dx_nu * I_sc
            J20_mu += phi_nu[n] * dx_nu * (P2 * I_sc + 3*(mu_m**2 - 1) * Q_sc)

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
    S[(2,0)] = (1 - epsilon) * J20 * H2

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

# Compute emergent Stokes parameters to check if Q vanishes
I_emerge = np.zeros(N_mu)
Q_emerge = np.zeros(N_mu)

for m in range(N_mu):
    mu_m = mu[m]
    # Tensor basis functions (paper Eq. 8-9)
    T_I = 0.5 * (3*mu_m**2 - 1)           # (1/√2) × P2, but paper uses unnormalized
    T_Q = 1.5 * (1 - mu_m**2)             # (3/(2√2)) × (1-μ²), but paper uses unnormalized
    
    # Emergent intensity and Q from source tensor components
    # At surface (τ → 0): I ≈ S_I, Q ≈ S_Q
    I_emerge[m] = S[(0,0)][-1] + T_I * S[(2,0)][-1]
    Q_emerge[m] = T_Q * S[(2,0)][-1]

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
plt.title("Anisotropy (Tensor formulation with frequency grid)")
plt.grid()
plt.savefig("anisotropy_tensor.png", dpi=300, bbox_inches='tight')
plt.show()


