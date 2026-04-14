import numpy as np
import matplotlib.pyplot as plt
from numba import jit

# --- Grids ---
N_tau = 50
tau = np.logspace(-4, 4, N_tau)  # surface → bottom

N_mu = 8
mu, w = np.polynomial.legendre.leggauss(N_mu)

N_nu = 41
x = np.linspace(-5, 5, N_nu)

# --- Doppler profile ---
phi = np.exp(-x**2) / np.sqrt(np.pi)
phi /= np.trapz(phi, x)

# --- Physical parameters ---
epsilon = 1e-4
B = 1.0  # isothermal

# --- Fields ---
I = np.zeros((N_tau, N_mu, N_nu))
Q = np.zeros((N_tau, N_mu, N_nu))

S_I = np.ones(N_tau) * B
S_Q = np.zeros(N_tau)

# --- Formal solver ---
def formal_solver(S_I, S_Q):
    I_new = np.zeros_like(I)
    Q_new = np.zeros_like(Q)

    for m in range(N_mu):
        mu_m = mu[m]
        if abs(mu_m) < 1e-10:
            continue

        if mu_m > 0:  # upward rays
            I_new[-1, m, :] = B
            Q_new[-1, m, :] = 0.0

            for i in reversed(range(N_tau - 1)):
                dtau = (tau[i+1] - tau[i]) / mu_m
                e = np.exp(-dtau)

                I_new[i, m, :] = I_new[i+1, m, :] * e + S_I[i] * (1 - e)
                Q_new[i, m, :] = Q_new[i+1, m, :] * e + S_Q[i] * (1 - e)

        else:  # downward rays
            I_new[0, m, :] = 0.0
            Q_new[0, m, :] = 0.0

            for i in range(1, N_tau):
                dtau = (tau[i] - tau[i-1]) / abs(mu_m)
                e = np.exp(-dtau)

                I_new[i, m, :] = I_new[i-1, m, :] * e + S_I[i] * (1 - e)
                Q_new[i, m, :] = Q_new[i-1, m, :] * e + S_Q[i] * (1 - e)

    return I_new, Q_new

# --- Iteration ---
n_iter = 60

for it in range(n_iter):
    I, Q = formal_solver(S_I, S_Q)

    # --- Compute radiation field tensors ---
    w3d = w.reshape(1, N_mu, 1)
    mu3d = mu.reshape(1, N_mu, 1)

    # J^0_0
    J00 = 0.5 * np.sum(w3d * I, axis=1)  # (tau, nu)

    # J^0_2 (CORRECT irreducible tensor normalization)
    J02 = (1 / (4 * np.sqrt(2))) * np.sum(
        w3d * (3 * mu3d**2 - 1) * I,
        axis=1
    )

    # --- Frequency integration with Doppler profile ---
    J00_int = np.trapz(J00 * phi, x, axis=1)
    J02_int = np.trapz(J02 * phi, x, axis=1)

    # --- Update source functions (consistent normalization!) ---
    S_I_new = (1 - epsilon) * J00_int + epsilon * B
    S_Q_new = (1 - epsilon) * J02_int

    # enforce isotropy at bottom
    S_Q_new[-1] = 0.0

    # --- Convergence ---
    err = max(
        np.max(np.abs(S_I_new - S_I)),
        np.max(np.abs(S_Q_new - S_Q))
    )

    print(f"Iter {it}, error = {err:.3e}")

    S_I, S_Q = S_I_new, S_Q_new

    if err < 1e-5:
        break

# --- Final anisotropy ---
anisotropy = J02_int / (J00_int + 1e-12)

# --- Plot ---
plt.figure(figsize=(6,5))
plt.semilogx(tau, anisotropy, '-o')

plt.xlabel("Optical depth τ")
plt.ylabel(r"$J^0_2 / J^0_0$")
plt.title("Radiation Field Anisotropy (Correct Normalization)")

plt.grid()
plt.show()

# 2nd version, GS

# --- Grids ---
N_tau = 50
tau = np.logspace(-4, 4, N_tau)

N_mu = 12
mu, w = np.polynomial.legendre.leggauss(N_mu)

N_nu = 81
x = np.linspace(-5, 5, N_nu)

# --- Doppler profile ---
phi = np.exp(-x**2)/np.sqrt(np.pi)
phi /= np.trapz(phi, x)

# --- Physical parameters ---
epsilon = 1e-4
B = 1.0  # isothermal

# --- Fields ---
I = np.zeros((N_tau, N_mu, N_nu))
Q = np.zeros((N_tau, N_mu, N_nu))
S_I = np.ones(N_tau) * B
S_Q = np.zeros(N_tau)

# --- Precompute Legendre polynomial P2 ---
P2 = (3*mu**2 - 1)

# --- Formal solver using short-characteristics (full pass) ---
def formal_solver_full(S_I, S_Q):
    I_new = np.zeros_like(I)
    Q_new = np.zeros_like(Q)

    for m in range(N_mu):
        mu_m = mu[m]
        if abs(mu_m) < 1e-12:
            continue

        if mu_m > 0:  # upward
            I_new[-1, m, :] = B
            Q_new[-1, m, :] = 0.0
            for i in reversed(range(N_tau - 1)):
                dtau = (tau[i+1] - tau[i])/mu_m
                e = np.exp(-dtau)
                I_new[i, m, :] = I_new[i+1, m, :]*e + S_I[i]*(1-e)
                Q_new[i, m, :] = Q_new[i+1, m, :]*e + S_Q[i]*(1-e)

        else:  # downward
            I_new[0, m, :] = 0.0
            Q_new[0, m, :] = 0.0
            for i in range(1, N_tau):
                dtau = (tau[i] - tau[i-1])/abs(mu_m)
                e = np.exp(-dtau)
                I_new[i, m, :] = I_new[i-1, m, :]*e + S_I[i]*(1-e)
                Q_new[i, m, :] = Q_new[i-1, m, :]*e + S_Q[i]*(1-e)

    return I_new, Q_new

# --- Compute radiation moments ---
def compute_J(I):
    w3d = w.reshape(1, N_mu, 1)
    J00 = 0.5*np.sum(w3d * I, axis=1)
    J02 = (1/(4*np.sqrt(2))) * np.sum(w3d * P2.reshape(1,N_mu,1) * I, axis=1)
    J00_int = np.trapz(J00*phi, x, axis=1)
    J02_int = np.trapz(J02*phi, x, axis=1)
    return J00_int, J02_int

# --- Optimized Gauss–Seidel iteration ---
n_iter = 50
for it in range(n_iter):
    S_I_old = S_I.copy()
    S_Q_old = S_Q.copy()

    # Full top→bottom + bottom→top sweep
    I, Q = formal_solver_full(S_I, S_Q)

    # Radiation moments
    J00_int, J02_int = compute_J(I)

    # Update source functions in place (Gauss–Seidel)
    S_I = (1-epsilon)*J00_int + epsilon*B
    S_Q = (1-epsilon)*J02_int
    S_Q[-1] = 0.0  # isotropic bottom

    # Convergence
    err = max(np.max(np.abs(S_I-S_I_old)), np.max(np.abs(S_Q-S_Q_old)))
    print(f"Iter {it}, error = {err:.3e}")
    if err < 1e-6:
        break

# --- Anisotropy J02/J00 ---
anisotropy = J02_int / (J00_int + 1e-12)

# --- Plot ---
plt.figure(figsize=(6,5))
plt.semilogx(tau, anisotropy, '-o')
plt.xlabel("Optical depth τ")
plt.ylabel(r"$J^0_2 / J^0_0$")
plt.title("Radiation Field Anisotropy (Optimized Gauss–Seidel)")
plt.grid()
plt.show()

# 6th version, tensor components with short-characteristics
# --- Grid ---
N_tau = 100
tau = np.logspace(-4, 8, N_tau)

N_mu = 12
mu, w = np.polynomial.legendre.leggauss(N_mu)

# --- Physics ---
epsilon = 1e-4
B = 1.0
omega = 1.0  # relaxation

# --- Tensor source components ---
S00 = np.ones(N_tau) * B
S20 = np.zeros(N_tau)

# --- Angular factors ---
P2 = (3 * mu**2 - 1)

# --- Intensities ---
I = np.zeros((N_tau, N_mu))
Q = np.zeros((N_tau, N_mu))

# --- Short Characteristics ---
@jit(nopython=True)
def short_characteristics(tau, S, mu, I_boundary):
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

    return I

# --- Iteration ---
n_iter = 50

for it in range(n_iter):
    S00_old = S00.copy()
    S20_old = S20.copy()

    J00 = np.zeros(N_tau)
    J02 = np.zeros(N_tau)

    for m in range(N_mu):
        mu_m = mu[m]
        w_m = w[m]

        # --- Build angle-dependent source functions ---
        S_I_mu = S00 + (1/np.sqrt(2)) * (3*mu_m**2 - 1) * S20
        S_Q_mu = (3/(2*np.sqrt(2))) * (1 - mu_m**2) * S20

        # --- Boundary conditions ---
        I_boundary = B if mu_m > 0 else 0.0
        Q_boundary = 0.0

        # --- Solve transfer ---
        I_sc = short_characteristics(tau, S_I_mu, mu_m, I_boundary)
        Q_sc = short_characteristics(tau, S_Q_mu, mu_m, Q_boundary)

        I[:, m] = I_sc
        Q[:, m] = Q_sc

        # --- Radiation field tensors ---
        J00 += 0.5 * w_m * I_sc
        J02 += (1/(4*np.sqrt(2))) * w_m * (
    P2[m] * I_sc + 3*(mu_m**2 - 1) * Q_sc)

    # --- Statistical equilibrium ---
    S00_new = (1 - epsilon) * J00 + epsilon * B
    S20_new = (1 - epsilon) * J02

    # --- Relaxation ---
    S00 = S00 + omega * (S00_new - S00)
    S20 = S20 + omega * (S20_new - S20)

    # Bottom boundary: isotropy
    S20[-1] = 0.0

    # --- Convergence ---
    err = max(np.max(np.abs(S00 - S00_old)),
              np.max(np.abs(S20 - S20_old)))

    print(f"Iter {it}: error = {err:.3e}")

    if err < 1e-6:
        break

# --- Anisotropy ---
anisotropy = J02 / (J00 + 1e-12)

# --- Plot ---
plt.figure(figsize=(6,5))
plt.semilogx(tau, anisotropy, '-o')
plt.xlabel("Optical depth τ")
plt.ylabel(r"$J^0_2 / J^0_0$")
plt.title("Anisotropy (Tensor formulation)")
plt.grid()
plt.legend()
plt.show()

# --- Source function polarization ---
mu_targets = [0.1, 0.5, 0.9]
# find closest indices
mu_idx = [np.argmin(np.abs(mu - mt)) for mt in mu_targets]
SQ_over_SI = {}
for idx in mu_idx:
    mu_m = mu[idx]

    S_I_mu = S00 + (1/np.sqrt(2)) * (3*mu_m**2 - 1) * S20
    S_Q_mu = (3/(2*np.sqrt(2))) * (1 - mu_m**2) * S20

    SQ_over_SI[mu_m] = S_Q_mu / (S_I_mu + 1e-12)

plt.figure(figsize=(6,5))

for mu_m, ratio in SQ_over_SI.items():
    plt.semilogx(tau, ratio, label=f"μ = {mu_m:.2f}")

plt.xlabel("Optical depth τ")
plt.ylabel(r"$S_Q / S_I$")
plt.title("Source Function Polarization")
plt.legend()
plt.grid()
plt.show()




# -----------------------
# GRID
# -----------------------
N_tau = 100
tau = np.logspace(-4, 8, N_tau)

N_mu = 12
mu, w_mu = np.polynomial.legendre.leggauss(N_mu)

# -----------------------
# PHYSICS
# -----------------------
epsilon = 1e-4
B = 1.0
omega = 1.0

# -----------------------
# TENSOR STORAGE
# -----------------------
def init_tensor_source(N):
    S = {}
    S[(0,0)] = np.ones(N) * B
    S[(2,0)] = np.zeros(N)

    # placeholders for future Q ≠ 0
    for q in ['1c','1s','2c','2s']:
        S[(2,q)] = np.zeros(N)

    return S

S = init_tensor_source(N_tau)

# -----------------------
# --- Short Characteristics ---
@jit(nopython=True)
def short_characteristics(tau, S, mu, I_boundary):
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

    return I

# -----------------------
# TENSOR SOURCE BUILDER
# (CURRENTLY AXISYMMETRIC)
# -----------------------
def build_source(mu_m, S):

    # Only Q=0 active for now
    S_I = (
        S[(0,0)]
        + (1/np.sqrt(2)) * (3*mu_m**2 - 1) * S[(2,0)]
    )

    S_Q = (
        (3/(2*np.sqrt(2))) * (1 - mu_m**2) * S[(2,0)]
    )

    return S_I, S_Q

# -----------------------
# RADIATION FIELD TENSORS
# -----------------------
def update_J(mu_m, w_m, I_sc, Q_sc, J):

    P2 = (3*mu_m**2 - 1)

    J[(0,0)] += 0.5 * w_m * I_sc

    J[(2,0)] += (1/(4*np.sqrt(2))) * w_m * (
        P2 * I_sc + 3*(mu_m**2 - 1) * Q_sc
    )

# -----------------------
# ITERATION LOOP
# -----------------------
n_iter = 50

for it in range(n_iter):

    S_old = {k: v.copy() for k, v in S.items()}

    J = {k: np.zeros(N_tau) for k in S.keys()}

    I = np.zeros((N_tau, N_mu))
    Q = np.zeros((N_tau, N_mu))

    for m in range(N_mu):

        mu_m = mu[m]
        w_m = w_mu[m]

        # --- build tensor source → direction-dependent sources
        S_I, S_Q = build_source(mu_m, S)

        # --- boundary conditions
        I_boundary = B if mu_m > 0 else 0.0
        Q_boundary = 0.0

        # --- solve transfer
        I_sc = short_characteristics(tau, S_I, mu_m, I_boundary)
        Q_sc = short_characteristics(tau, S_Q, mu_m, Q_boundary)

        I[:, m] = I_sc
        Q[:, m] = Q_sc

        # --- update tensors
        update_J(mu_m, w_m, I_sc, Q_sc, J)

    # -----------------------
    # STATISTICAL EQUILIBRIUM
    # -----------------------
    for k in J.keys():
        if k == (0,0):
            S[(0,0)] = (1 - epsilon) * J[(0,0)] + epsilon * B
        else:
            S[k] = (1 - epsilon) * J[k]

    # bottom boundary condition
    S[(2,0)][-1] = 0.0

    # -----------------------
    # convergence
    # -----------------------
    err = max(
        np.max(np.abs(S[(0,0)] - S_old[(0,0)])),
        np.max(np.abs(S[(2,0)] - S_old[(2,0)]))
    )

    print(f"Iter {it}: error = {err:.3e}")

    if err < 1e-6:
        break

# -----------------------
# ANISOTROPY
# -----------------------
anisotropy = S[(2,0)] / (S[(0,0)] + 1e-12)

# --- Plot ---
plt.figure(figsize=(6,5))
plt.semilogx(tau, anisotropy, '-o')
plt.xlabel("Optical depth τ")
plt.ylabel(r"$J^0_2 / J^0_0$")
plt.title("Anisotropy (Tensor formulation)")
plt.grid()
plt.legend()
plt.show()

# --- Source function polarization ---
mu_targets = [0.1, 0.5, 0.9]
# find closest indices
mu_idx = [np.argmin(np.abs(mu - mt)) for mt in mu_targets]
SQ_over_SI = {}
for idx in mu_idx:
    mu_m = mu[idx]

    S_I_mu = S00 + (1/np.sqrt(2)) * (3*mu_m**2 - 1) * S20
    S_Q_mu = (3/(2*np.sqrt(2))) * (1 - mu_m**2) * S20

    SQ_over_SI[mu_m] = S_Q_mu / (S_I_mu + 1e-12)

plt.figure(figsize=(6,5))

for mu_m, ratio in SQ_over_SI.items():
    plt.semilogx(tau, ratio, label=f"μ = {mu_m:.2f}")

plt.xlabel("Optical depth τ")
plt.ylabel(r"$S_Q / S_I$")
plt.title("Source Function Polarization")
plt.legend()
plt.grid()
plt.show()

# Adding azimuth grid and Stokes U !!
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

# -----------------------
# INITIALIZE TENSORS
# -----------------------
def init_tensor(N):
    S = {}
    S[(0,0)] = np.ones(N) * B
    S[(2,0)] = np.zeros(N)

    for q in ['1c','1s','2c','2s']:
        S[(2,q)] = np.zeros(N)

    return S

S = init_tensor(N_tau)

# --- Short Characteristics ---
@jit(nopython=True)
def short_characteristics(tau, S, mu, I_boundary):
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

    return I

# -----------------------
# TENSOR BUILDER
# -----------------------
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
            S[k] = (1 - epsilon) * J[k]

    # boundary condition
    S[(2,0)][-1] = 0.0

    # -----------------------
    # CONVERGENCE
    # -----------------------
    err = max(np.max(np.abs(S[k] - S_old[k])) for k in S.keys())

    print(f"Iter {it}: error = {err:.3e}")

    if err < 1e-6:
        break

# -----------------------
# DIAGNOSTIC
# -----------------------
anisotropy = S[(2,0)] / (S[(0,0)] + 1e-12)

# --- Plot ---
plt.figure(figsize=(6,5))
plt.semilogx(tau, anisotropy, '-o')
plt.xlabel("Optical depth τ")
plt.ylabel(r"$J^0_2 / J^0_0$")
plt.title("Anisotropy (Tensor formulation)")
plt.grid()
plt.legend()
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
ratio = np.max(np.abs(S[(2,0)] / (S[(0,0)] + 1e-12)))
print(f"Max anisotropy ratio S20/S00 = {ratio:.3e}")

# Check orthogonality numerically
test = 0.0
for m in range(N_mu):
    for k in range(N_chi):
        T = compute_tensors(mu[m], chi[k])
        w = w_mu[m] * w_chi / (4*np.pi)

        test += w * T[('I','1c')] * T[('I','1s')]

print(f"Orthogonality test (1c vs 1s): {test:.3e}")

plt.plot(tau, S[(2,0)]  , label='S20')
plt.xscale('log')
plt.legend()
plt.show()

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
Gamma = 1.0  # try 0, 1, 10

def hanle_factor(Gamma):
    return 1/5 + (4/5)/(1 + Gamma**2)

H2 = hanle_factor(Gamma)

# -----------------------
# INITIALIZE TENSORS
# -----------------------
def init_tensor(N):
    S = {}
    S[(0,0)] = np.ones(N) * B
    S[(2,0)] = np.zeros(N)

    for q in ['1c','1s','2c','2s']:
        S[(2,q)] = np.zeros(N)

    return S

S = init_tensor(N_tau)

# --- Short Characteristics ---
@jit(nopython=True)
def short_characteristics(tau, S, mu, I_boundary):
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

    return I

# -----------------------
# TENSOR COMPUTATION
# -----------------------
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

                J[q] += w * (
                    T[('I',label)] * I_sc +
                    T[('Q',label)] * Q_sc +
                    T[('U',label)] * U_sc
                )

    # -----------------------
    # STATISTICAL EQUILIBRIUM (HANLE)
    # -----------------------
    for k in S.keys():
        if k == (0,0):
            S[k] = (1 - epsilon) * J[k] + epsilon * B
        elif k[0] == 2:
            S[k] = (1 - epsilon) * H2 * J[k]

    # boundary condition
    S[(2,0)][-1] = 0.0

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
anisotropy = S[(2,0)] / (S[(0,0)] + 1e-12)
# --- Plot ---
plt.figure(figsize=(6,5))
plt.semilogx(tau, anisotropy, '-o')
plt.xlabel("Optical depth τ")
plt.ylabel(r"$J^0_2 / J^0_0$")
plt.title("Anisotropy (Tensor formulation)")
plt.grid()
plt.legend()
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
ratio = np.max(np.abs(S[(2,0)] / (S[(0,0)] + 1e-12)))
print(f"Max anisotropy ratio S20/S00 = {ratio:.3e}")

# Check orthogonality numerically
test = 0.0
for m in range(N_mu):
    for k in range(N_chi):
        T = compute_tensors(mu[m], chi[k])
        w = w_mu[m] * w_chi / (4*np.pi)

        test += w * T[('I','1c')] * T[('I','1s')]

print(f"Orthogonality test (1c vs 1s): {test:.3e}")