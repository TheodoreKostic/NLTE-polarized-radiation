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
def short_characteristics(S, tau, mu_m, I_boundary):
    ND = len(S)
    I_sc = np.zeros(ND)

    if mu_m > 0:
        start, end, step = ND-1, -1, -1
    else:
        start, end, step = 0, ND, 1

    I_sc[start] = I_boundary

    for d in range(start + step, end - step, step):
        delta_u = (tau[d-step] - tau[d]) / mu_m
        delta_d = (tau[d] - tau[d+step]) / mu_m

        expd = np.exp(-delta_u)

        if delta_u <= 0.01:
            w0 = delta_u*(1.-delta_u/2.+delta_u**2/6.)
            w1 = delta_u**2*(0.5-delta_u/3.+delta_u**2/8.)
            w2 = delta_u**3*(1./3.-delta_u/4.+delta_u**2/10.)
        else:
            w0 = 1.0 - expd
            w1 = w0 - delta_u * expd
            w2 = 2.0 * w1 - delta_u**2 * expd

        psi0 = w0 + (w1*(delta_u/delta_d - delta_d/delta_u)
                     - w2*(1.0/delta_d + 1.0/delta_u)) / (delta_u + delta_d)
        psiu = (w2/delta_u + w1*delta_d/delta_u) / (delta_u + delta_d)
        psid = (w2/delta_d - w1*delta_u/delta_d) / (delta_u + delta_d)

        I_sc[d] = (I_sc[d-step]*expd +
                   psiu*S[d-step] +
                   psi0*S[d] +
                   psid*S[d+step])

    # last point (linear)
    d = end - step
    delta_u = (tau[d-step] - tau[d]) / mu_m
    expd = np.exp(-delta_u)

    if delta_u < 0.01:
        expd = 1.0 - delta_u + delta_u**2/2. - delta_u**3/6.
        psi0 = delta_u/2. - delta_u**2/6. + delta_u**3/24.
        psiu = delta_u/2. - delta_u**2/3. + delta_u**3/8.
    else:
        psi0 = 1.0 - (1.0 - expd)/delta_u
        psiu = -expd + (1.0 - expd)/delta_u

    I_sc[d] = I_sc[d-step]*expd + psiu*S[d-step] + psi0*S[d]

    return I_sc

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
        I_sc = short_characteristics(S_I_mu, tau, mu_m, I_boundary)
        Q_sc = short_characteristics(S_Q_mu, tau, mu_m, Q_boundary)

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


# ----------------------------
# GRID
# ----------------------------
N_tau = 100
tau = np.logspace(-4, 8, N_tau)

N_mu = 16
mu, w = np.polynomial.legendre.leggauss(N_mu)

P2 = 0.5 * (3*mu**2 - 1)

N_x = 201
x = np.linspace(-5, 5, N_x)
phi = np.exp(-x**2) / np.sqrt(np.pi)
phi /= np.trapz(phi, x)

# ----------------------------
# PARAMETERS (Fig. 10 style)
# ----------------------------
B = 1.0
mu_target = 0.1

epsilon = 1e-4
D2_list = [0.0, 0.1, 1.0, 10.0]


# ----------------------------
# 2nd-order Short Characteristics (UNCHANGED)
# ----------------------------
@jit(nopython=True)
def sc_2nd_order(tau, S, mu, I_boundary):
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

    return I, L

# ----------------------------
# SOLVER (FIXED ANISOTROPY DRIVER)
# ----------------------------
def solve(epsilon, D2, n_iter=25):

    # ✔ KEY FIX: introduce vertical anisotropy (BREAKS SYMMETRY)
    S00 = B * (1.0 + 0.2 * np.exp(-tau / 2.0))
    S20 = np.ones(N_tau) * 1e-6

    imu = np.argmin(np.abs(mu - mu_target))

    relax = 0.3

    for it in range(n_iter):

        J00 = np.zeros(N_tau)
        J20 = np.zeros(N_tau)

        # ----------------------------
        # RADIATION FIELD
        # ----------------------------
        for ix in range(N_x):

            J0_loc = np.zeros(N_tau)
            J2_loc = np.zeros(N_tau)

            for m in range(N_mu):

                mu_m = mu[m]

                S_I = S00 + (1/np.sqrt(2)) * P2[m] * S20

                I, _ = sc_2nd_order(
                    tau,
                    S_I,
                    mu_m,
                    B if mu_m < 0 else 0.0
                )

                I = np.clip(I, 0.0, 10.0 * B)

                J0_loc += w[m] * I
                J2_loc += w[m] * 0.5 * (3*mu_m**2 - 1) * I

            J00 += phi[ix] * J0_loc
            J20 += phi[ix] * J2_loc

        # ----------------------------
        # SOURCE FUNCTIONS
        # ----------------------------
        S00_new = (1 - epsilon) * J00 + epsilon * B
        S20_new = ((1 - epsilon) * J20) / (1.0 + D2)

        # relaxation (stability)
        S00 = (1 - relax) * S00 + relax * S00_new
        S20 = (1 - relax) * S20 + relax * S20_new

    # ----------------------------
    # EMERGENT Q/I
    # ----------------------------
    QI = np.zeros(N_x)

    for ix in range(N_x):

        S_I = S00 + (1/np.sqrt(2)) * P2[imu] * S20

        I, _ = sc_2nd_order(
            tau,
            S_I,
            mu_target,
            B
        )

        Q0 = (3/(2*np.sqrt(2))) * (1 - mu_target**2) * S20[0]

        QI[ix] = Q0 / (I[0] + 1e-12)

    return QI


# ----------------------------
# PLOT
# ----------------------------
plt.figure(figsize=(6,5))

for D2 in D2_list:
    QI = solve(epsilon, D2)
    plt.plot(x, 100*QI, label=f"D²={D2}")

plt.xlabel("x (Doppler units)")
plt.ylabel("Q/I (%)")
plt.legend()
plt.grid()
plt.show()