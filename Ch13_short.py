import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from functions_prt import short_characteristics

# ---------------------------
# Model / numerical settings
# ---------------------------
N_tau = 120
tau = np.logspace(-6, 2.5, N_tau)   # optical depth grid (top->bottom)
N_mu = 16
mu, wmu = np.polynomial.legendre.leggauss(N_mu)
# keep only upward (mu>0) for emergent later; but quadrature uses -1..1
epsilon = 1e-4     # thermalization parameter (small -> scattering dominated)
B = 1.0            # Planck function (isothermal)
max_iter = 200
tol = 1e-6

def compute_J_from_S(S):
    # compute angle-averaged intensity J(τ) = 0.5 ∫_{-1}^{1} I(µ) dµ
    ND = len(tau)
    J = np.zeros(ND)
    for k, mu_k in enumerate(mu):
        I_boundary = B if mu_k > 0 else 0.0
        out = short_characteristics(tau, S, mu_k, I_boundary, ali=False)
        I_mu = out[0]        # short_characteristics returns stack((I,L))
        J += 0.5 * wmu[k] * I_mu
    return J

# ---------------------------
# NLTE iteration (Lambda iteration)
# ---------------------------
def solve_nlte():
    # initial guess source function S = B
    S = np.ones_like(tau) * B
    for it in range(max_iter):
        J = compute_J_from_S(S)
        S_new = (1.0 - epsilon) * J + epsilon * B
        err = np.max(np.abs(S_new - S))
        S = S_new
        if it % 10 == 0:
            print(f"Iter {it}, err={err:.3e}")
        if err < tol:
            print(f"Converged iter {it}, err={err:.3e}")
            break
    else:
        print("Warning: NLTE iteration did not converge to tolerance")
    return S

# ---------------------------
# Get emergent intensity I(0, mu>0)
# ---------------------------
def emergent_I_surface(S):
    # compute I at top (index 0) for all upward mu>0 using repository short-characteristics
    I_surf = []
    mu_up = mu[mu > 0]
    for mu_val in mu_up:
        out = short_characteristics(tau, S, mu_val, I_boundary=B, ali=False)
        I_mu = out[0]
        I_surf.append(I_mu[0])   # intensity at top along that mu
    return np.array(mu_up), np.array(I_surf)

# ---------------------------
# Disk-integration as seen from a point at height h
# (book geometry: integrate over visible spherical cap)
# ---------------------------
def compute_J0_J2_from_disk(I_mu_func, mu_grid_disk, h, npsi=500):
    # I_mu_func(mu_disk) should give surface intensity at given µ (center-to-limb)
    # gamma: angular radius of the visible cap: sin γ = 1/(1+h)
    sin_gamma = 1.0 / (1.0 + h)
    gamma = np.arcsin(sin_gamma)
    # integrate over cap: parameterize by polar angle psi in [0,gamma], azimuth phi in [0,2π)
    psi = np.linspace(0.0, gamma, npsi)
    dpsi = psi[1] - psi[0]
    # perform integral over azimuth analytically (factor 2π)
    J0 = 0.0
    J2 = 0.0
    for psi_i in psi:
        cos_psi = np.cos(psi_i)
        sin_psi = np.sin(psi_i)
        # geometry to compute mu (cosθ) at disk element:
        # following book: direction from surface element to P -> mu = ((1+h) - cosψ)/r
        x = sin_psi
        y = cos_psi - (1.0 + h)
        r = np.sqrt(x*x + y*y)
        mu_elem = ((1.0 + h) - cos_psi) / r
        mu_elem = np.clip(mu_elem, -1.0, 1.0)
        I_elem = I_mu_func(mu_elem)
        # weight for surface element projection (see derivation in Chap.12)
        weight = (mu_elem * sin_psi) / (r*r)
        J0 += I_elem * weight * (2.0 * np.pi) * dpsi
        J2 += I_elem * (0.5*(3*mu_elem*mu_elem - 1.0)) * weight * (2.0 * np.pi) * dpsi
    # normalization: J0 and J2 here already integrate intensity over solid angle; to use book conventions we
    # return J0, J2 and anisotropy w = J2 / (sqrt(2) * J0)
    return J0, J2

# utility to interpolate I(mu) from computed emergent intensities
from scipy.interpolate import interp1d

def make_I_mu_function(mu_up, I_up):
    # normalize to center intensity I(mu=1)
    f = interp1d(mu_up, I_up, kind='cubic', bounds_error=False, fill_value=(I_up[0], I_up[-1]))
    return lambda mu_query: float(f(np.clip(mu_query, mu_up.min(), mu_up.max())))

# ---------------------------
# Book/analytic limb-darkening (Allen) and analytic-check
# ---------------------------
def limb_darkening_mu(mu, u1=0.95, u2=-0.20):
    # Quadratic limb-darkening: I(µ)/I(1) = 1 - u1*(1-µ) - u2*(1-µ)**2
    return 1.0 - u1*(1.0 - mu) - u2*(1.0 - mu)**2

def compute_J0_J2_book(u_mu_func, h, npsi=1000):
    # same geometry as compute_J0_J2_from_disk but use analytic I(mu) from u_mu_func
    sin_gamma = 1.0 / (1.0 + h)
    gamma = np.arcsin(sin_gamma)
    psi = np.linspace(0.0, gamma, npsi)
    dpsi = psi[1] - psi[0]
    J0 = 0.0
    J2 = 0.0
    for psi_i in psi:
        cos_psi = np.cos(psi_i)
        sin_psi = np.sin(psi_i)
        x = sin_psi
        y = cos_psi - (1.0 + h)
        r = np.sqrt(x*x + y*y)
        mu_elem = ((1.0 + h) - cos_psi) / r
        mu_elem = np.clip(mu_elem, -1.0, 1.0)
        I_elem = u_mu_func(mu_elem)
        weight = (mu_elem * sin_psi) / (r*r)
        J0 += I_elem * weight * (2.0 * np.pi) * dpsi
        J2 += I_elem * (0.5*(3*mu_elem*mu_elem - 1.0)) * weight * (2.0 * np.pi) * dpsi
    return J0, J2

# ---------------------------
# Polarization formula from Eq. (13.23) (frequency-integrated)
# I = τ [ J0 + 1/(2√2) (3 sin^2 δ - 1) J2 ]
# Q = τ [ 3/(2√2) cos^2 δ J2 ]
# thus p_Q = Q / I  (τ cancels)
# ---------------------------
def pQ_from_J(J0, J2, delta_rad):
    I_fac = J0 + (1.0/(2.0*np.sqrt(2.0))) * (3.0 * (np.sin(delta_rad)**2) - 1.0) * J2
    Q_fac = (3.0/(2.0*np.sqrt(2.0))) * (np.cos(delta_rad)**2) * J2
    return Q_fac / I_fac

# ---------------------------
# Main execution
# ---------------------------
if __name__ == "__main__":
    print("Solving NLTE two-level atom (frequency-integrated, CRD) ...")
    S = solve_nlte()

    # get emergent I(mu) at surface
    mu_up, I_up = emergent_I_surface(S)

    # normalize intensities to disk-center I(mu=1)
    I_up /= I_up.max()

    # build I(mu) function
    I_mu_fun = make_I_mu_function(mu_up, I_up)

    # ---------------------------
    # Analytic (Allen) check: compare J0,J2 and anisotropy w for sample heights
    # ---------------------------
    u1, u2 = 0.95, -0.20
    u_func = lambda mu: limb_darkening_mu(mu, u1, u2)
    heights = [0.001, 0.01, 0.05, 0.1]
    print("Analytic (Allen) vs NLTE anisotropy check (w = J2/(sqrt(2)*J0)):")
    for h in heights:
        J0_book, J2_book = compute_J0_J2_book(u_func, h)
        J0_nlte, J2_nlte = compute_J0_J2_from_disk(I_mu_fun, mu_up, h)
        w_book = J2_book / (np.sqrt(2.0) * J0_book) if J0_book!=0 else np.nan
        w_nlte  = J2_nlte  / (np.sqrt(2.0) * J0_nlte)  if J0_nlte!=0 else np.nan
        print(f"h={h:.4f}: w_book={w_book:.6e}, w_nlte={w_nlte:.6e}, diff={w_nlte-w_book:.2e}")

    # h' (projected height) grid
    hprime_vals = np.linspace(0.0, 0.18, 181)
    delta_deg = [-30, -20, -10, 0, 10, 20, 30]
    delta_rad = np.radians(delta_deg)

    plt.figure(figsize=(8,5))
    for d_rad, d_deg in zip(delta_rad, delta_deg):
        p_vals = []
        for hprime in hprime_vals:
            # get true height h from book relation: (h + R) cos δ = h' + R  (R=1)
            h = (hprime + 1.0)/np.cos(d_rad) - 1.0
            J0, J2 = compute_J0_J2_from_disk(I_mu_fun, mu_up, h)
            p = pQ_from_J(J0, J2, d_rad)
            p_vals.append(p)
        plt.plot(hprime_vals, p_vals, label=f"δ={d_deg}°")

    plt.xlabel(r"$h'/R$")
    plt.ylabel(r"$p_Q$")
    plt.title("Fig.13.2 — NLTE-based recreation")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    # save figure instead of showing interactively
    plt.savefig('fig13_nlte.png', dpi=200)
    print('Plot saved to fig13_nlte.png')