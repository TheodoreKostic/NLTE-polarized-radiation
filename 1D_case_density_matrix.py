import numpy as np
import matplotlib.pyplot as plt
from numba import jit
from functions_prt import *

# 1D case, Q = 0 and K = {0, 2}

# Tau grid
N_tau = 150
tau = np.logspace(-4, 8, N_tau)

# Angle grid
N_mu = 8
mu, w_mu = np.polynomial.legendre.leggauss(N_mu)

# Frequency grid
N_nu = 81
nu = np.linspace(-5, 5, N_nu)
dnu = nu[1] - nu[0]
phi_nu = doppler_profile(nu)
phi_nu /= np.trapezoid(phi_nu, x=nu)

# Parameters
epsilon = 1e-4  # thermalization parameter
B = 1.0 # Planck function (constant in this case)

# --- Hanle parameter ---
Gamma = 1.0  
H2 = hanle_factor(Gamma)
H2 = 1.0  # No Hanle effect for this test

S = {
    (0,0): np.ones(N_tau) * B,
    (2,0): np.zeros(N_tau)
}

n_iter = 150
for it in range(n_iter):

    S_old = {k: v.copy() for k, v in S.items()}

    J00 = np.zeros(N_tau)
    J20 = np.zeros(N_tau)

    # -------------------------
    # Formal solution + moments
    # -------------------------

    for m in range(N_mu):

        mu_m = mu[m]
        T_I = T20(mu_m)

        S_I = S[(0,0)] + T_I * S[(2,0)]

        I_boundary = B if mu_m > 0 else 0.0

        for n in range(N_nu):

            phi = phi_nu[n]
            tau_nu = tau * phi

            I = short_characteristics(
                tau_nu, S_I, mu_m, I_boundary
            )
            
            W = w_mu[m] * phi * dnu / 2.0

            J00 += W * I
            J20 += W * T_I * I

    # -------------------------
    # Source update (irreducible SE)
    # -------------------------

    S[(0,0)] = (1 - epsilon) * J00 + epsilon * B
    S[(2,0)] = (1 - epsilon) * H2 * J20 * 0.5

    # -------------------------
    # Convergence check
    # -------------------------
    diff = max(
        np.max(np.abs(S[(0,0)] - S_old[(0,0)])),
        np.max(np.abs(S[(2,0)] - S_old[(2,0)]))
    )

    if diff < 1e-6:
        print(f"Converged in {it} iterations")
        break


# anisotropy
anisotropy_depth = J20 / (J00 + 1e-12)

I_surf = []
Q_surf = []

for m in range(N_mu):

    T_I = T20(mu[m])
    T_Q = T2Q(mu[m])

    I_emerge = S[(0,0)] + T_I * S[(2,0)]
    Q_emerge = T_Q * S[(2,0)]

    I_surf.append(I_emerge[-1])
    Q_surf.append(Q_emerge[-1])

I_surf = np.array(I_surf)
Q_surf = np.array(Q_surf)

Q_over_I = Q_surf / (I_surf + 1e-12)

plt.figure(figsize=(12, 12))
plt.subplot(2, 1, 1)
plt.semilogx(tau, anisotropy_depth, '-*', label='Anisotropy')
plt.xlabel('Optical Depth')
plt.ylabel(r"$J^2_0/J^0_0$")
plt.title('Anisotropy vs Optical Depth')
plt.grid()
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(mu, Q_over_I, '-o', label='Emergent Q/I')
plt.xlabel("Frequency")
plt.ylabel("Q/I")
plt.title("Emergent Polarization")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("anisotropy_and_polarization.png", dpi=300, bbox_inches='tight')

# Check Q
max_Q = np.max(np.abs(Q_surf))
print("Max |Q| =", max_Q)

Q_over_I = Q_surf / (I_surf + 1e-12)
max_QI = np.max(np.abs(Q_over_I))
print("Max |Q/I| =", max_QI)
print("--------------------------------------")
print("Max S20:", np.max(np.abs(S[(2,0)])))
print("--------------------------------------")

print("Max J20:", np.max(np.abs(J20)))
print("Max J00:", np.max(np.abs(J00)))
print('--------------------------------------')

