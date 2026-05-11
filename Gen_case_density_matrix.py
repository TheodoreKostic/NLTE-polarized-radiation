import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from numba import jit
from functions_prt import *

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

        _, T20 = irreducible_weights(mu_m)

        I_store = np.zeros((N_tau, N_nu))

        # scalar source
        S_I = build_scalar_source(S00, S20, mu_m)

        for n in range(N_nu):

            phi = phi_nu[n]

            tau_eff = tau * phi

            if mu_m > 0:
                I_boundary = B
            else:
                I_boundary = 0.0

            I_sc, _ = short_characteristics(
                tau_eff,
                S_I,
                mu_m,
                I_boundary
            )

            I_store[:,n] = I_sc

        # frequency integration
        I_nu = np.trapezoid(
            phi_nu[None,:] * I_store,
            x_nu,
            axis=1
        )

        # angular quadrature
        J00 += 0.5 * w_m * I_nu

        J20 += 0.5 * w_m * T20 * I_nu

    return J00, J20

def stokes_vs_tau(mu_m, chi_l, S00, S2):

    # -----------------------------------------
    # Frequency-dependent storage
    # -----------------------------------------

    I_tau_nu = np.zeros((N_tau, N_nu))
    Q_tau_nu = np.zeros((N_tau, N_nu))
    U_tau_nu = np.zeros((N_tau, N_nu))

    # -----------------------------------------
    # Irreducible tensors
    # -----------------------------------------

    TI, TQ, TU = T2_emissivity_tensors(mu_m, chi_l)

    # -----------------------------------------
    # Build source functions
    # -----------------------------------------

    SI = S00.copy()

    SQ = np.zeros(N_tau)
    SU = np.zeros(N_tau)

    for i in range(5):

        SI += np.real(TI[i] * S2[i])

        SQ += np.real(TQ[i] * S2[i])

        SU += np.real(TU[i] * S2[i])

    # -----------------------------------------
    # Formal solutions for each frequency
    # -----------------------------------------

    for n in range(N_nu):

        tau_eff = tau * phi_nu[n]

        if mu_m > 0:
            I_boundary = B
        else:
            I_boundary = 0.0

        I_sc, _ = short_characteristics(
            tau_eff,
            np.real(SI),
            mu_m,
            I_boundary
        )

        Q_sc, _ = short_characteristics(
            tau_eff,
            np.real(SQ),
            mu_m,
            0.0
        )

        U_sc, _ = short_characteristics(
            tau_eff,
            np.real(SU),
            mu_m,
            0.0
        )

        I_tau_nu[:, n] = I_sc
        Q_tau_nu[:, n] = Q_sc
        U_tau_nu[:, n] = U_sc

    # -----------------------------------------
    # Integrate over frequency
    # -----------------------------------------

    I_tau = np.trapezoid(
        phi_nu[None,:] * I_tau_nu,
        x_nu,
        axis=1
    )

    Q_tau = np.trapezoid(
        phi_nu[None,:] * Q_tau_nu,
        x_nu,
        axis=1
    )

    U_tau = np.trapezoid(
        phi_nu[None,:] * U_tau_nu,
        x_nu,
        axis=1
    )

    return I_tau, Q_tau, U_tau

def build_scalar_source(S00, S20, mu):

    _, T20 = irreducible_weights(mu)

    return S00 + T20 * S20

def emergent_stokes(mu_m, S00, S2):

    I_nu = np.zeros(N_nu)
    Q_nu = np.zeros(N_nu)
    U_nu = np.zeros(N_nu)

    for n in range(N_nu):

        phi = phi_nu[n]
        tau_eff = tau * phi

        TI, TQ, TU = T2_emissivity_tensors(mu_m, 0.0)

        SI_src = np.real(S00 + TI[2] * S2[2])

        SQ_src = np.zeros_like(S00)
        SU_src = np.zeros_like(S00)

        for i, Q in enumerate(Qvals):

            SQ_src += np.real(TQ[i] * S2[i])
            SU_src += np.real(TU[i] * S2[i])

        if mu_m > 0:
            I_boundary = B
        else:
            I_boundary = 0.0

        I_sc, _ = short_characteristics(tau_eff, SI_src, mu_m, I_boundary)
        Q_sc, _ = short_characteristics(tau_eff, SQ_src, mu_m, 0.0)
        U_sc, _ = short_characteristics(tau_eff, SU_src, mu_m, 0.0)

        I_nu[n] = I_sc[0]
        Q_nu[n] = Q_sc[0]
        U_nu[n] = U_sc[0]

    I = np.trapezoid(phi_nu * I_nu, x_nu)
    Q = np.trapezoid(phi_nu * Q_nu, x_nu)
    U = np.trapezoid(phi_nu * U_nu, x_nu)

    return I, Q, U

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
# ============================================================
# INITIAL CONDITIONS
# ============================================================

S00 = np.ones(N_tau) * B
S20 = np.zeros(N_tau)

# Hanle matrix (constant)
Gamma = (1.3996e6 * g_u * B_field) / (A_ul + Gamma_I)

# ============================================================
# SCALAR NLTE ITERATION
# ============================================================

n_iter = 2000
tol = 1e-6
W2 = 1.0

for it in range(n_iter):

    S00_old = S00.copy()
    S20_old = S20.copy()

    # compute radiation tensors
    J00, J20 = compute_radiation_tensor(S00, S20)

    # update source tensors
    S00 = (1 - epsilon) * J00 + epsilon * B

    S20 = (1 - epsilon) * W2 * J20

    # upper boundary condition
    S20[-1] = 0.0

    err = max(
        np.max(np.abs(S00 - S00_old)),
        np.max(np.abs(S20 - S20_old))
    )

    print(f"Iter {it}: err = {err:.3e}")

    if err < tol:
        print("Converged.")
        break


# ============================================================
# BUILD FULL TENSOR VECTOR
# ============================================================

S2 = np.zeros((5, N_tau), dtype=complex)

# only Q=0 initially populated
S2[qindex(0), :] = S20.copy()

theta_B = np.radians(60.0)
chi_B = np.radians(30.0)
Gamma = 1.0 # test

S2_hanle = apply_hanle_effect(
    S2,
    Gamma,
    theta_B,
    chi_B
)

S2m2 = S2_hanle[qindex(-2)]
S2m1 = S2_hanle[qindex(-1)]
S20  = S2_hanle[qindex(0)]
S21  = S2_hanle[qindex(1)]
S22  = S2_hanle[qindex(2)]

I_out = []
Q_out = []
U_out = []

for m in range(N_mu):

    I, Q, U = emergent_stokes(
        mu[m],
        S00,
        S2_hanle
    )

    I_out.append(I)
    Q_out.append(Q)
    U_out.append(U)

I_out = np.array(I_out)
Q_out = np.array(Q_out)
U_out = np.array(U_out)

mask = mu > 0.0
plt.plot(mu[mask], Q_out[mask] / I_out[mask] * 100, label="Q/I")
plt.plot(mu[mask], U_out[mask] / I_out[mask] * 100, label="U/I")
plt.xlabel(r"$\mu$")
plt.ylabel("Polarization (%)")
plt.legend()
plt.grid()
plt.title("Scattering Polarization with Hanle Effect")
plt.savefig("scattering_polarization.png", dpi=300)
plt.show()

anisotropy = J20 / J00
print("Max J20: ", np.max(J20))
print("Min J20: ", np.min(J20))
print("Surface anisotropy: ", anisotropy[0])
plt.figure(figsize=(6,5))
plt.plot(tau, anisotropy, '-o')
plt.xscale('log')
plt.xlabel(r"$\tau$")
plt.ylabel(r"$J^2_0 / J^0_0$")
plt.title("Radiation Anisotropy")
plt.grid()
plt.savefig("anisotropy.png", dpi=300)
plt.show()

# -------------------------------------------------------------------
mu_test = 0.1
chi_test = 0.0
I_tau, Q_tau, U_tau = stokes_vs_tau(
    mu_test,
    chi_test,
    S00,
    S2_hanle
)
plt.figure(figsize=(7,5))
plt.semilogx(tau, I_tau, label='I')
plt.semilogx(tau, Q_tau, label='Q')
plt.semilogx(tau, U_tau, label='U')
plt.xlabel(r'Optical depth $\tau$')
plt.ylabel('Stokes parameters')
plt.legend()
plt.grid()
plt.title(f'Stokes vs tau for mu={mu_test:.2f}, chi={chi_test:.2f} rad')
plt.savefig("stokes_vs_tau.png", dpi=300)
plt.show()

plt.figure(figsize=(7,5))
plt.semilogx(tau, Q_tau / I_tau, label='Q/I')
plt.semilogx(tau, U_tau / I_tau, label='U/I')
plt.xlabel(r'Optical depth $\tau$')
plt.ylabel('Fractional polarization')
plt.legend()
plt.grid()
plt.title(f'Fractional polarization vs tau for mu={mu_test:.2f}, chi={chi_test:.2f} rad')
plt.savefig("fractional_polarization_vs_tau.png", dpi=300)
plt.show()

#---------------------------------------------------
# Visualization for more mu and chi
mu_test = [0.01, 0.1, 0.25, 0.5, 0.75, 1.0]
chi_test = [0.0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi]

fig, axes = plt.subplots(
    2, 3,
    figsize=(15,10),
    sharex=True,
    sharey=True
)

axes = axes.flatten()

for k, mu_m in enumerate(mu_test):

    ax = axes[k]

    for chi_m in chi_test:

        I_tau, Q_tau, U_tau = stokes_vs_tau(
            mu_m,
            chi_m,
            S00,
            S2_hanle
        )

        label = rf'$\chi={chi_m/np.pi:.2f}\pi$'

        ax.semilogx(
            tau,
            Q_tau/(I_tau + 1e-12),
            label=label
        )

    ax.set_title(rf'$\mu={mu_m}$')
    ax.grid()

axes[0].legend(fontsize=8)
fig.supxlabel(r'Optical depth $\tau$')
fig.supylabel(r'$Q/I$')
plt.tight_layout()
plt.show()

fig, axes = plt.subplots(
    2, 3,
    figsize=(15,10),
    sharex=True,
    sharey=True
)

axes = axes.flatten()

for k, mu_m in enumerate(mu_test):

    ax = axes[k]

    for chi_m in chi_test:

        I_tau, Q_tau, U_tau = stokes_vs_tau(
            mu_m,
            chi_m,
            S00,
            S2_hanle
        )

        label = rf'$\chi={chi_m/np.pi:.2f}\pi$'

        ax.semilogx(
            tau,
            U_tau/(I_tau + 1e-12),
            label=label
        )

    ax.set_title(rf'$\mu={mu_m}$')
    ax.grid()

axes[0].legend(fontsize=8)
fig.supxlabel(r'Optical depth $\tau$')
fig.supylabel(r'$U/I$')
plt.tight_layout()
plt.show()

angle = 0.5 * np.arctan2(U_tau, Q_tau)

plt.semilogx(
    tau,
    np.degrees(angle)
)
plt.xlabel(r'Optical depth $\tau$')
plt.ylabel('Polarization angle (degrees)')
plt.grid()
plt.title(f'Polarization angle vs tau')
plt.savefig("polarization_angle_vs_tau.png", dpi=300)
plt.show()  

Qmap = np.zeros((len(mu_test), N_tau))

for i, mu_m in enumerate(mu_test):

    I_tau, Q_tau, U_tau = stokes_vs_tau(
        mu_m,
        0.0,
        S00,
        S2_hanle
    )

    Qmap[i] = Q_tau/(I_tau + 1e-12)

plt.pcolormesh(
    tau,
    mu_test,
    Qmap,
    shading='auto'
)
plt.xscale('log')
plt.xlabel(r'$\tau$')
plt.ylabel(r'$\mu$')
plt.colorbar(label='Q/I')
plt.title('Q/I map vs tau and mu')
plt.savefig("Q_over_I_map.png", dpi=300)
plt.show()


# Use I as a start approximation for the next try

# Test problem sa J u jednoj tacki - kako se S, I, Q menjaju u jednoj tacki
# Slicno kao sto se tretira u proturebancama - primeri u Polarization in Spectral Lines, Landi Degl'Innocenti & Landolfi, 2004 
# Iskoristiti liniju stroncijuma Sr 6887 A, koja je poznata po svojoj linearnoj polarizaciji u solarnom spektru.


# Moze se krenuti iz iskonvergiranog skalarnog resenja, pa iz toga odrediti ostalo sa LI
# >>> Resiti skalarnu JPZ, odrediti J00, J20 iz kojih onda odrediti S20, S22, 
# pa onda iz toga odrediti I, Q, U (dovoljno je koristiti LI, ali moze i bez iteracija)
# 
