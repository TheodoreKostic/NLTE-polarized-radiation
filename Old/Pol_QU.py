import numpy as np
import matplotlib.pyplot as plt
from numba import jit
from functions_prt import hanle_factor, init_tensor, short_characteristics, compute_tensors, doppler_profile

# ---------------------------------------------------
# Hanle implementation ------------------------------
# ---------------------------------------------------
# Still no frequency dependence, but we have the full tensor formalism for polarization, including the Hanle effect.


# -----------------------
# GRID
# -----------------------
N_tau = 100
tau = np.logspace(-4, 4, N_tau)

N_mu = 8
mu, w_mu = np.polynomial.legendre.leggauss(N_mu)

N_chi = 8
chi = np.linspace(0, 2*np.pi, N_chi, endpoint=False)
w_chi = 2*np.pi / N_chi



# -----------------------
# PHYSICS
# -----------------------
epsilon = 1e-4  # thermalization parameter
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

S_I_avg = np.zeros(N_tau)
S_Q_avg = np.zeros(N_tau)

for m in range(N_mu):
    for k in range(N_chi):
        w = w_mu[m] * w_chi / (4*np.pi)

        T = compute_tensors(mu[m], chi[k])

        S_I_avg += w * (
            S[(0,0)]
        )

        S_Q_avg += w * (
            T[('Q','1c')] * S[(2,'1c')] +
            T[('Q','1s')] * S[(2,'1s')] +
            T[('Q','2c')] * S[(2,'2c')] +
            T[('Q','2s')] * S[(2,'2s')]
        )

plt.figure(figsize=(24,18))
plt.tight_layout()
plt.subplot(3,3,1)
plt.semilogx(tau, S_I_avg, '-o', label=r"$S^I$")   
plt.xlabel("Optical depth τ")
plt.ylabel("Source function for Stokes I")
plt.title("Source function (Tensor formulation), gray atmosphere")
plt.grid()
plt.legend()

plt.subplot(3,3,2)
plt.semilogx(tau, S_Q_avg, '-o', label=r"$S^Q$")
plt.xlabel("Optical depth τ")
plt.ylabel(r"$S^Q$")
plt.title("Source function for Stokes Q (Tensor formulation), gray atmosphere")
plt.grid()
plt.legend()

plt.subplot(3,3,3)
plt.semilogx(tau, S_Q_avg/S_I_avg, '-o', label=r"$S^Q/S^I$")
plt.xlabel("Optical depth τ")
plt.ylabel(r"$S^Q/S^I$")
plt.title("Polarization fraction (Tensor formulation), gray atmosphere")
plt.grid()
plt.legend()

plt.subplot(3,3,4)
plt.semilogx(tau, J[(0,0)], '-o', label=r"$J^0_0$")
plt.xlabel("Optical depth τ")
plt.ylabel(r"$J^0_0$")
plt.title("Mean intensity (Tensor formulation), gray atmosphere")
plt.grid()
plt.legend()

plt.subplot(3,3,5)
plt.semilogx(tau, J[(2,0)], '-o', label=r"$J^0_2$")
plt.xlabel("Optical depth τ")
plt.ylabel(r"$J^0_2$")
plt.title("Alignment (Tensor formulation), gray atmosphere")
plt.grid()
plt.legend()    

plt.subplot(3,3,6)
plt.semilogx(tau, anisotropy, '-o')
plt.xlabel("Optical depth τ")
plt.ylabel(r"$J^0_2 / J^0_0$")
plt.title("Anisotropy (Tensor formulation), gray atmosphere")
plt.grid()
plt.legend()

m0 = 0
k0 = 0

plt.subplot(3,3,7)
plt.semilogx(tau, I[:,m0,k0], '-o', label=r"$I$")
plt.xlabel("Optical depth τ")
plt.ylabel(r"$I$")
plt.title("Stokes I (Tensor formulation), gray atmosphere")
plt.grid()
plt.legend()

plt.subplot(3,3,8)
plt.semilogx(tau, Q[:,m0,k0], '-o', label=r"$Q$")
plt.xlabel("Optical depth τ")
plt.ylabel(r"$Q$")
plt.title("Stokes Q (Tensor formulation), gray atmosphere")
plt.grid()
plt.legend()    

plt.subplot(3,3,9)
plt.semilogx(tau, Q[:,m0,k0]/I[:,m0,k0], '-o', label=r"$Q/I$")
plt.xlabel("Optical depth τ")
plt.ylabel(r"$Q/I$")
plt.title("Polarization ratio (Tensor formulation), gray atmosphere")
plt.grid()
plt.legend()

plt.savefig("hanle_tensor_gray_results.png", dpi=300)


# Next, we want to add frequency dependence and see how the Hanle effect modifies the line profile. 
# This will require adding a frequency grid 
# and including the Doppler profile in the source function and the scattering integral.

# -----------------------
# FREQUENCY GRID
# -----------------------
N_nu = 41
x = np.linspace(-5, 5, N_nu)   # Doppler units
phi_nu = doppler_profile(x)
w_nu = np.ones(N_nu) / N_nu  # simple quadrature
# -----------------------
# MAIN ITERATION
# -----------------------
n_iter = 50

for it in range(n_iter):

    S_old = {k: v.copy() for k,v in S.items()}
    J = {k: np.zeros(N_tau) for k in S.keys()}

    # allocate intensities
    I = np.zeros((N_tau, N_mu, N_chi, N_nu))
    Q = np.zeros_like(I)
    U = np.zeros_like(I)

    # -----------------------
    # LOOP OVER ANGLES FIRST
    # -----------------------
    for m in range(N_mu):
        mu_m = mu[m]

        for k in range(N_chi):
            chi_k = chi[k]

            # --- angular tensors (computed ONCE per ray) ---
            T = compute_tensors(mu_m, chi_k)

            # -----------------------
            # BUILD SOURCE (freq independent)
            # -----------------------
            S_I = S[(0,0)].copy()
            S_Q = np.zeros(N_tau)
            S_U = np.zeros(N_tau)

            for q in [(2,0),(2,'1c'),(2,'1s'),(2,'2c'),(2,'2s')]:
                label = q[1]

                S_I += T[('I',label)] * S[q]
                S_Q += T[('Q',label)] * S[q]
                S_U += T[('U',label)] * S[q]

            # boundary condition
            I_boundary = B if mu_m > 0 else 0.0

            # -----------------------
            # LOOP OVER FREQUENCY
            # -----------------------
            for n in range(N_nu):

                phi = phi_nu[n]

                tau_nu = tau * phi

                # solve RT
                I_sc = short_characteristics(tau_nu, S_I, mu_m, I_boundary)
                Q_sc = short_characteristics(tau_nu, S_Q, mu_m, 0.0)
                U_sc = short_characteristics(tau_nu, S_U, mu_m, 0.0)

                # store
                I[:,m,k,n] = I_sc
                Q[:,m,k,n] = Q_sc
                U[:,m,k,n] = U_sc

                # -----------------------
                # ACCUMULATE J
                # -----------------------
                w = w_mu[m] * w_chi * w_nu[n] / (4*np.pi)

                J[(0,0)] += w * I_sc

                for q in [(0,0),(2,0),(2,'1c'),(2,'1s'),(2,'2c'),(2,'2s')]:
                    label = q[1]

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
        else:
            S[k] = (1 - epsilon) * H2 * J[k]

    # S[(2,0)][-1] = 0.0

    # -----------------------
    # CONVERGENCE
    # -----------------------
    err = max(np.max(np.abs(S[k] - S_old[k])) for k in S.keys())

    print(f"Iter {it}: error = {err:.3e}")

    if err < 1e-6:
        break


# anisotropy
anisotropy_depth = J[(2,0)] / (J[(0,0)] + 1e-12)

S_I_avg = np.zeros(N_tau)
S_Q_avg = np.zeros(N_tau)

for m in range(N_mu):
    for k in range(N_chi):
        w = w_mu[m] * w_chi / (4*np.pi)

        T = compute_tensors(mu[m], chi[k])

        S_I_avg += w * (
            S[(0,0)]
        )

        S_Q_avg += w * (
            T[('Q','1c')] * S[(2,'1c')] +
            T[('Q','1s')] * S[(2,'1s')] +
            T[('Q','2c')] * S[(2,'2c')] +
            T[('Q','2s')] * S[(2,'2s')]
        )

# 1. Check non-zero tensor components
for key in [(2,'1c'), (2,'1s'), (2,'2c'), (2,'2s')]:
    max_val = np.max(np.abs(S[key]))
    print(f"Max S{key} = {max_val:.3e}")

# 2. Check Stokes U
max_U = np.max(np.abs(U))
print(f"Max U = {max_U:.3e}")


plt.figure(figsize=(20,12.5))
plt.subplot(3,3,1)
plt.semilogx(tau, S_I_avg, '-o', label=r"$S^I$")   
plt.xlabel("Optical depth τ")
plt.ylabel("Source function for Stokes I")
plt.title("Source function frequency-dependent")
plt.grid()
plt.legend()

plt.subplot(3,3,2)
plt.semilogx(tau, S_Q_avg, '-o', label=r"$S^Q$")
plt.xlabel("Optical depth τ")
plt.ylabel(r"$S^Q$")
plt.title("Source function for Stokes Q frequency-dependent")
plt.grid()
plt.legend()

plt.subplot(3,3,3)
plt.semilogx(tau, S_Q_avg/S_I_avg, '-o', label=r"$S^Q/S^I$")
plt.xlabel("Optical depth τ")
plt.ylabel(r"$S^Q/S^I$")
plt.title("Polarization fraction frequency-dependent")
plt.grid()
plt.legend()

plt.subplot(3,3,4)
plt.semilogx(tau, J[(0,0)], '-o', label=r"$J^0_0$")
plt.xlabel("Optical depth τ")
plt.ylabel(r"$J^0_0$")
plt.title("Mean intensity frequency-dependent")
plt.grid()
plt.legend()

plt.subplot(3,3,5)
plt.semilogx(tau, J[(2,0)], '-o', label=r"$J^0_2$")
plt.xlabel("Optical depth τ")
plt.ylabel(r"$J^0_2$")
plt.title("Alignment frequency-dependent")
plt.grid()
plt.legend()    

plt.subplot(3,3,6)
plt.semilogx(tau, anisotropy_depth, '-o')
plt.xlabel("Optical depth τ")
plt.ylabel(r"$J^0_2 / J^0_0$")
plt.title("Anisotropy frequency-dependent")
plt.grid()
plt.legend()

m0 = 0
k0 = 0
n0 = N_nu // 2

plt.subplot(3,3,7)
plt.semilogx(tau, I[:, m0, k0, n0], '-o', label=r"$I$")
plt.xlabel("Optical depth τ")
plt.ylabel(r"$I$")
plt.title("Stokes I frequency-dependent")
plt.grid()
plt.legend()

plt.subplot(3,3,8)
plt.semilogx(tau, Q[:, m0, k0, n0], '-o', label=r"$Q$")
plt.xlabel("Optical depth τ")
plt.ylabel(r"$Q$")
plt.title("Stokes Q frequency-dependent")
plt.grid()
plt.legend()    

plt.subplot(3,3,9)
plt.semilogx(tau, Q[:, m0, k0, n0]/(I[:, m0, k0, n0] + 1e-12), '-o', label=r"$Q/I$")
plt.xlabel("Optical depth τ")
plt.ylabel(r"$Q/I$")
plt.title("Polarization ratio frequency-dependent")
plt.grid()
plt.legend()

plt.savefig("hanle_tensor_freq_results.png", dpi=300)

I_em = np.mean(I[0,:,:,:], axis=(0,1))

plt.figure(figsize=(8,5))
plt.plot(x, I_em)
plt.title("Emergent intensity vs frequency")
plt.grid()
plt.xlabel("Frequency (Doppler units)")
plt.ylabel("Emergent intensity")
plt.savefig("hanle_emergent_profile.png", dpi=300)