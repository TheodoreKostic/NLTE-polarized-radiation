import numpy as np
import matplotlib.pyplot as plt
from numba import jit
from functions_prt import hanle_factor, init_tensor, short_characteristics, compute_tensors

# ---------------------------------------------------
# Hanle implementation ------------------------------
# ---------------------------------------------------

# -----------------------
# GRID
# -----------------------
N_tau = 100
tau = np.logspace(-4, 8, N_tau)

N_mu = 12
mu, w_mu = np.polynomial.legendre.leggauss(N_mu)

N_chi = 8
chi = np.linspace(0, 2*np.pi, N_chi, endpoint=False)
w_chi = 2*np.pi / N_chi

# -----------------------
# PHYSICS
# -----------------------
epsilon = 1e-4
B = 1.0
omega = 1.0

# --- Hanle parameter ---
Gamma = 1.0  

H2 = hanle_factor(Gamma)

# -----------------------
# INITIALIZE TENSORS
# -----------------------

S = init_tensor(N_tau)

# -----------------------
# ITERATION
# -----------------------
n_iter = 50

for it in range(n_iter):

    S_old = {k: v.copy() for k,v in S.items()}
    J = {k: np.zeros(N_tau) for k in S.keys()}

    I = np.zeros((N_tau, N_mu, N_chi))
    Q = np.zeros((N_tau, N_mu, N_chi))
    U = np.zeros((N_tau, N_mu, N_chi))

    for m in range(N_mu):
        mu_m = mu[m]

        for k in range(N_chi):
            chi_k = chi[k]

            T = compute_tensors(mu_m, chi_k)

            # --- build source ---
            S_I = S[(0,0)].copy()
            S_Q = np.zeros(N_tau)
            S_U = np.zeros(N_tau)

            for q in [(2,0),(2,'1c'),(2,'1s'),(2,'2c'),(2,'2s')]:
                label = q[1]

                S_I += T[('I',label)] * S[q]
                S_Q += T[('Q',label)] * S[q]
                S_U += T[('U',label)] * S[q]

            # --- boundary ---
            I_boundary = B if mu_m > 0 else 0.0

            # --- solve ---
            I_sc = short_characteristics(tau, S_I, mu_m, I_boundary)
            Q_sc = short_characteristics(tau, S_Q, mu_m, 0.0)
            U_sc = short_characteristics(tau, S_U, mu_m, 0.0)

            I[:,m,k] = I_sc
            Q[:,m,k] = Q_sc
            U[:,m,k] = U_sc

            # --- accumulate J ---
            w = w_mu[m] * w_chi / (4*np.pi)

            J[(0,0)] += w * I_sc

            for q in [(2,0),(2,'1c'),(2,'1s'),(2,'2c'),(2,'2s')]:
                label = q[1]
                # add (0, 0) contribution to J[q]
                J[q] += w * (
                    T[('I',label)] * I_sc +
                    T[('Q',label)] * Q_sc +
                    T[('U',label)] * U_sc
                )

    # -----------------------
    # STATISTICAL EQUILIBRIUM 
    # -----------------------
    for k in S.keys():
        if k == (0,0):
            S[k] = (1 - epsilon) * J[k] + epsilon * B
        elif k[0] == 2:
            S[k] = (1 - epsilon) * H2 * J[k]

    # boundary condition
    S[(2,0)][-1] = 0.0 # print without this

    # -----------------------
    # CONVERGENCE
    # -----------------------
    err = max(np.max(np.abs(S[k] - S_old[k])) for k in S.keys())

    print(f"Iter {it}: error = {err:.3e}")

    if err < 1e-6:
        break

# -----------------------
# DIAGNOSTICS
# -----------------------
print("\n--- HANLE DIAGNOSTICS ---")
print(f"Gamma = {Gamma}")
print(f"Hanle factor H2 = {H2:.3f}")
print(f"Max S20 = {np.max(np.abs(S[(2,0)])):.3e}")

# anisotropy
anisotropy = J[(2,0)] / (J[(0,0)] + 1e-12)
# --- Plot ---
plt.figure(figsize=(6,5))
plt.semilogx(tau, anisotropy, '-o')
plt.xlabel("Optical depth τ")
plt.ylabel(r"$J^0_2 / J^0_0$")
plt.title("Anisotropy (Tensor formulation)")
plt.grid()
plt.show()

# 1. Check non-zero tensor components
for key in [(2,'1c'), (2,'1s'), (2,'2c'), (2,'2s')]:
    max_val = np.max(np.abs(S[key]))
    print(f"Max S{key} = {max_val:.3e}")

# 2. Check Stokes U
max_U = np.max(np.abs(U))
print(f"Max U = {max_U:.3e}")

# 3. Check chi symmetry (I should not depend on chi)
chi_variation = np.max(np.abs(I - np.mean(I, axis=2, keepdims=True)))
print(f"Max chi-variation in I = {chi_variation:.3e}")

# 4. Check Q symmetry (should also be chi-independent)
Q_variation = np.max(np.abs(Q - np.mean(Q, axis=2, keepdims=True)))
print(f"Max chi-variation in Q = {Q_variation:.3e}")

# 5. Compare with expected anisotropy relation
ratio = np.max(np.abs(J[(2,0)] / (J[(0,0)] + 1e-12)))
print(f"Max anisotropy ratio J20/J00 = {ratio:.3e}")

# Check orthogonality numerically
test = 0.0
for m in range(N_mu):
    for k in range(N_chi):
        T = compute_tensors(mu[m], chi[k])
        w = w_mu[m] * w_chi / (4*np.pi)

        test += w * T[('I','1c')] * T[('I','1s')]

print(f"Orthogonality test (1c vs 1s): {test:.3e}")

# Plot S_I
plt.figure(figsize=(6,5))
plt.semilogx(tau, S_I, '-o', label=r"$S^I$")   
plt.xlabel("Optical depth τ")
plt.ylabel("Source function")
plt.title("Source function (Tensor formulation)")
plt.grid()
plt.legend()

# Plot S_Q over tau
plt.figure(figsize=(6,5))
plt.semilogx(tau, S_Q, '-o', label=r"$S^Q$")
plt.xlabel("Optical depth τ")
plt.ylabel(r"$S^Q$")
plt.title("Anisotropy (Tensor formulation)")
plt.grid()
plt.legend()
plt.show()