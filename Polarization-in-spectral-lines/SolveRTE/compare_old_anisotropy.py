"""Compare the new SolveRTE machinery with the old scalar anisotropy code (Old/Anisotropy.py).

The old code solves an isothermal, gray (frequency-independent opacity), unpolarized
two-level problem with source function S = (1 - eps) J^0_0 + eps B, no radiation
entering from the top and I = B entering from the bottom. It reports
J^2_0 / J^0_0 versus optical depth with
    J^0_0 = 1/2 sum_m w_m I_m,      J^2_0 = 1/(4 sqrt 2) sum_m w_m (3 mu_m^2 - 1) I_m.

Three separate questions are tested:
  1. Conventions: the same intensity field gives the same J^0_0, J^2_0 from the old
     formulas and from grid_tensors.jkq_from_stokes (tolerance 1e-12).
  2. Transfer: the old first-order sweep, short characteristics, and cell-wise exact
     propagation with formal_solver.propagate_constant_slab (source constant per cell,
     equal to the mean of its two nodes), measured against a fine-grid reference.
  3. Angular grid: the default Grid.build quadrature versus the old 8-node Gauss-Legendre.

Both Lambda-iterations are the plain (unaccelerated) kind. eps = 1e-2 is used so that
they converge; with the old defaults (eps = 1e-4, 60 iterations) the old result is not
converged, which is also shown.

Run from this directory:  python compare_old_anisotropy.py
(writes plots_slab/compare_old_anisotropy.png)
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numba import jit

from atmosphere import Atmosphere1D, Grid, LineModel
from formal_solver import propagate_constant_slab
from grid_tensors import jkq_from_stokes

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots_slab")
B_PLANCK = 1.0
EPS = 1e-2
TAU_MIN, TAU_MAX = 1e-4, 1e3
N_LIST = (49, 97, 193, 385)   # depth-grid refinement levels
N_REF = 1537                  # reference grid; every level above is a subset of it
N_MU_OLD = 8


def check(name, error, tol):
    ok = np.isfinite(error) and error < tol
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {error:.2e} (tol {tol:.0e})")
    if not ok:
        raise AssertionError(f"{name}: error {error:.3e} not below {tol:.3e}")


# ---------------------------------------------------------------- old algorithm
def old_moments(I, mu, w):
    """J^0_0 and J^2_0 exactly as in Old/Anisotropy.py. I has shape [N, nmu]."""
    J00 = 0.5 * I @ w
    J20 = (1.0 / (4.0 * np.sqrt(2.0))) * I @ (w * (3.0 * mu**2 - 1.0))
    return J00, J20


def old_sweep(S, tau, mu):
    """Old formal solver (formal_solver_full) for one frequency: [N, nmu].

    The source of a cell is taken at the node the ray arrives at (first order).
    """
    N = len(tau)
    up, dn = mu > 0, mu < 0
    I = np.zeros((N, len(mu)))
    I[-1, up] = B_PLANCK
    for i in range(N - 2, -1, -1):
        e = np.exp(-(tau[i + 1] - tau[i]) / mu[up])
        I[i, up] = I[i + 1, up] * e + S[i] * (1 - e)
    for i in range(1, N):
        e = np.exp(-(tau[i] - tau[i - 1]) / np.abs(mu[dn]))
        I[i, dn] = I[i - 1, dn] * e + S[i] * (1 - e)
    return I


@jit(nopython=True)
def short_characteristics(tau, S, mu, I_boundary):
    """Parabolic short-characteristics sweep for one ray and scalar source."""
    ND = S.shape[0]
    begin = ND - 1
    end = -1
    step = -1
    if mu < 0:
        begin = 0
        end = ND
        step = 1

    I = np.zeros(ND)
    L = np.zeros(ND)
    I[begin] = I_boundary

    for d in range(begin + step, end - step, step):
        delta_u = (tau[d - step] - tau[d]) / mu
        delta_d = (tau[d] - tau[d + step]) / mu
        expd = np.exp(-delta_u)

        if delta_u <= 0.01:
            du = delta_u
            w0 = du * (
                1.0 - du / 2.0 + du**2 / 6.0 - du**3 / 24.0 + du**4 / 120.0
                - du**5 / 720.0 + du**6 / 5040.0 - du**7 / 40320.0 + du**8 / 362880.0
            )
            w1 = du**2 * (
                0.5 - du / 3.0 + du**2 / 8.0 - du**3 / 30.0 + du**4 / 144.0
                - du**5 / 840.0 + du**6 / 5760.0 - du**7 / 45360.0 + du**8 / 403200.0
            )
            w2 = du**3 * (
                1.0 / 3.0 - du / 4.0 + du**2 / 10.0 - du**3 / 36.0 + du**4 / 168.0
                - du**5 / 960.0 + du**6 / 6480.0 - du**7 / 50400.0 + du**8 / 443520.0
            )
        else:
            w0 = 1.0 - expd
            w1 = w0 - delta_u * expd
            w2 = 2.0 * w1 - delta_u**2 * expd

        psi0 = w0 + (
            w1 * (delta_u / delta_d - delta_d / delta_u)
            - w2 * (1.0 / delta_d + 1.0 / delta_u)
        ) / (delta_u + delta_d)
        psiu = (w2 / delta_u + w1 * delta_d / delta_u) / (delta_u + delta_d)
        psid = (w2 / delta_d - w1 * delta_u / delta_d) / (delta_u + delta_d)

        I[d] = I[d - step] * expd + psiu * S[d - step] + psi0 * S[d] + psid * S[d + step]
        L[d] = psi0

    d = end - step
    delta_u = (tau[d - step] - tau[d]) / mu
    expd = np.exp(-delta_u)
    if delta_u < 0.01:
        expd = 1.0 - delta_u + delta_u**2 / 2.0 - delta_u**3 / 6.0
        psi0 = delta_u / 2.0 - delta_u**2 / 6.0 + delta_u**3 / 24.0
        psiu = delta_u / 2.0 - delta_u**2 / 3.0 + delta_u**3 / 8.0
    else:
        psi0 = 1.0 - (1.0 - expd) / delta_u
        psiu = -expd + (1.0 - expd) / delta_u

    I[d] = I[d - step] * expd + psiu * S[d - step] + psi0 * S[d]
    L[d] = psi0
    return I, L


def short_sweep(S, tau, mu):
    """Sweep all ordinates with the parabolic short-characteristics solver."""
    I = np.zeros((len(tau), len(mu)))
    for m, mu_m in enumerate(mu):
        boundary = B_PLANCK if mu_m > 0 else 0.0
        I[:, m], _ = short_characteristics(tau, S, mu_m, boundary)
    return I


def old_iteration(tau, mu, w, eps, n_iter, tol=0.0):
    """The old Lambda-iteration; returns S, J00, J20, I, number of iterations used."""
    S = np.full(len(tau), B_PLANCK)
    for it in range(n_iter):
        I = old_sweep(S, tau, mu)
        J00, J20 = old_moments(I, mu, w)
        S_new = (1 - eps) * J00 + eps * B_PLANCK
        err = np.max(np.abs(S_new - S))
        S = S_new
        if err < tol:
            break
    I = old_sweep(S, tau, mu)
    J00, J20 = old_moments(I, mu, w)
    return S, J00, J20, I, it + 1


# ------------------------------------------------------- new machinery
def make_grid(tau, mu, w_mu):
    """A Grid carrying the angular quadrature; one depth point per tau value."""
    n = len(tau)
    atm = Atmosphere1D(
        z=-np.asarray(tau), T=np.full(n, 6000.0), B=np.zeros(n),
        theta_B=np.zeros(n), chi_B=np.zeros(n), v_los=np.zeros(n),
        D2=np.zeros(n),
    )
    line = LineModel(
        6.5e14, 2e8, 1.0, 1.0, 1.0, 0.0, 87.6,
        n_l=np.ones(n), C_ul=np.zeros(n),
    )
    phi = 2 * np.pi * np.arange(4) / 4
    return Grid(atm, line, np.asarray(mu), np.asarray(w_mu), phi, np.full(4, 2 * np.pi / 4),
                np.array([-1.0, 1.0]), np.array([1.0, 1.0]))


def stokes_from_intensity(grid, I):
    """Unpolarized Stokes field [N, 4, nnu, nmu, nphi] from a gray intensity [N, nmu]."""
    N, nmu = I.shape
    S = np.zeros((N, 4, len(grid.nu), nmu, len(grid.phi)))
    S[:, 0] = I[:, None, :, None]
    return S


def jkq00_20(grid, I):
    """J^0_0 and J^2_0 from jkq_from_stokes (flat profile normalized to 1)."""
    profile = np.full((grid.atm.N, len(grid.nu)), 1.0 / grid.w_nu.sum())
    J = jkq_from_stokes(grid, stokes_from_intensity(grid, I), profile=profile)
    return J[:, 0, 2].real, J[:, 2, 2].real


def moment_weights(grid):
    """Linear weights (wJ00[m], wJ20[m]) with J = sum_m w[m] I[:, m], measured through jkq_from_stokes."""
    nmu = len(grid.mu)
    w00, w20 = np.zeros(nmu), np.zeros(nmu)
    for m in range(nmu):
        I = np.zeros((grid.atm.N, nmu))
        I[:, m] = 1.0
        a, b = jkq00_20(grid, I)
        w00[m], w20[m] = a[0], b[0]
    return w00, w20


def cell_coefficients(tau, mu):
    """Exact cell propagation I_out = a I_in + b S_cell for eta_I = 1, via propagate_constant_slab.

    Returns arrays a, b of shape [N - 1, nmu] for the cell between nodes i and i + 1.
    """
    dist = np.diff(tau)[:, None] / np.abs(mu)[None, :]
    a, b = np.zeros_like(dist), np.zeros_like(dist)
    coeff = np.array([1.0, 0, 0, 0, 0, 0, 0])
    unit = np.array([1.0, 0, 0, 0])
    zero = np.zeros(4)
    for i in range(dist.shape[0]):
        for m in range(dist.shape[1]):
            a[i, m] = propagate_constant_slab(zero, coeff, dist[i, m], unit)[0]
            b[i, m] = propagate_constant_slab(unit, coeff, dist[i, m], zero)[0]
    return a, b


def new_sweep(S, mu, a, b):
    """Formal solution with exact cell propagation; the cell source is the mean of its nodes."""
    N = len(S)
    Sc = 0.5 * (S[:-1] + S[1:])
    up, dn = mu > 0, mu < 0
    I = np.zeros((N, len(mu)))
    I[-1, up] = B_PLANCK
    for i in range(N - 2, -1, -1):
        I[i, up] = a[i, up] * I[i + 1, up] + b[i, up] * Sc[i]
    for i in range(1, N):
        I[i, dn] = a[i - 1, dn] * I[i - 1, dn] + b[i - 1, dn] * Sc[i - 1]
    return I


def new_iteration(tau, mu, w_mu, eps, tol=1e-11, max_iter=50000):
    """Lambda-iteration with the new propagator and jkq_from_stokes moments."""
    grid = make_grid(tau, mu, w_mu)
    w00, w20 = moment_weights(grid)
    a, b = cell_coefficients(tau, mu)
    S = np.full(len(tau), B_PLANCK)
    for it in range(max_iter):
        I = new_sweep(S, mu, a, b)
        S_new = (1 - eps) * (I @ w00) + eps * B_PLANCK
        err = np.max(np.abs(S_new - S))
        S = S_new
        if err < tol:
            break
    I = new_sweep(S, mu, a, b)
    J00, J20 = jkq00_20(grid, I)
    return S, J00, J20, I, it + 1, err < tol


def short_iteration(tau, mu, w_mu, eps, tol=1e-11, max_iter=50000):
    """Lambda-iteration using short characteristics and jkq_from_stokes moments."""
    grid = make_grid(tau, mu, w_mu)
    w00, _ = moment_weights(grid)
    S = np.full(len(tau), B_PLANCK)
    for it in range(max_iter):
        I = short_sweep(S, tau, mu)
        S_new = (1 - eps) * (I @ w00) + eps * B_PLANCK
        err = np.max(np.abs(S_new - S))
        S = S_new
        if err < tol:
            break
    I = short_sweep(S, tau, mu)
    J00, J20 = jkq00_20(grid, I)
    return S, J00, J20, I, it + 1, err < tol


def gauss_legendre(n):
    return np.polynomial.legendre.leggauss(n)


def solve_new(n, mu, w):
    tau = np.logspace(np.log10(TAU_MIN), np.log10(TAU_MAX), n)
    out = new_iteration(tau, mu, w, EPS)
    assert out[5], f"new Lambda-iteration did not converge for N={n}"
    return tau, out[1], out[2]


def solve_short(n, mu, w):
    tau = np.logspace(np.log10(TAU_MIN), np.log10(TAU_MAX), n)
    out = short_iteration(tau, mu, w, EPS)
    assert out[5], f"short-characteristics Lambda-iteration did not converge for N={n}"
    return tau, out[1], out[2]


def solve_old(n, mu, w):
    tau = np.logspace(np.log10(TAU_MIN), np.log10(TAU_MAX), n)
    _, J00, J20, _, _ = old_iteration(tau, mu, w, EPS, 200000, 1e-11)
    return tau, J00, J20


def main():
    os.makedirs(OUT, exist_ok=True)
    mu8, w8 = gauss_legendre(N_MU_OLD)

    # 1. Conventions: identical moments from an arbitrary intensity field.
    tau_c = np.logspace(np.log10(TAU_MIN), np.log10(TAU_MAX), N_LIST[0])
    rng = np.random.default_rng(0)
    I_rand = rng.random((len(tau_c), N_MU_OLD))
    g8 = make_grid(tau_c, mu8, w8)
    j00_o, j20_o = old_moments(I_rand, mu8, w8)
    j00_n, j20_n = jkq00_20(g8, I_rand)
    check("conventions: J^0_0 old formula vs jkq_from_stokes", np.max(np.abs(j00_o - j00_n)), 1e-12)
    check("conventions: J^2_0 old formula vs jkq_from_stokes", np.max(np.abs(j20_o - j20_n)), 1e-12)

    # 2. Cell propagation versus the closed form exp(-dtau/mu) for eta_I only.
    a, b = cell_coefficients(tau_c, mu8)
    e = np.exp(-np.diff(tau_c)[:, None] / np.abs(mu8)[None, :])
    check("cell propagator: a = exp(-dtau/mu)", np.max(np.abs(a - e)), 1e-13)
    check("cell propagator: b = 1 - exp(-dtau/mu)", np.max(np.abs(b - (1 - e))), 1e-13)
    constant_source = np.full(len(tau_c), 0.37)
    for mu_m in mu8:
        I_sc, _ = short_characteristics(tau_c, constant_source, mu_m, 0.37)
        check(
            f"short characteristics: constant-source ray mu={mu_m:.2f}",
            np.max(np.abs(I_sc - 0.37)),
            1e-12,
        )

    # 3. Refinement study (same 8 angles). Every grid is a subset of the reference grid.
    tau_ref, J00_ref, J20_ref = solve_new(N_REF, mu8, w8)
    w_ref = J20_ref / J00_ref
    scale_w = np.max(np.abs(w_ref))
    err = lambda x, y: np.max(np.abs(x - y))
    sols = {}
    print(
        f"{'N':>5} | {'old J00':>9} {'old w':>9} | {'SC J00':>9} {'SC w':>9} | "
        f"{'new J00':>9} {'new w':>9}   (max errors vs N={N_REF} reference)"
    )
    for n in N_LIST:
        step = (N_REF - 1) // (n - 1)
        _, jo0, jo2 = solve_old(n, mu8, w8)
        _, js0, js2 = solve_short(n, mu8, w8)
        _, jn0, jn2 = solve_new(n, mu8, w8)
        sols[n] = (jo0, jo2 / jo0, jn0, jn2 / jn0)
        r0, rw = J00_ref[::step], w_ref[::step]
        sols[n] += (err(jo0, r0), err(jo2 / jo0, rw), err(jn0, r0), err(jn2 / jn0, rw))
        sols[n] += (js0, js2 / js0, err(js0, r0), err(js2 / js0, rw))
        print(
            f"{n:5d} | {sols[n][4]:9.2e} {sols[n][5]:9.2e} | "
            f"{sols[n][10]:9.2e} {sols[n][11]:9.2e} | "
            f"{sols[n][6]:9.2e} {sols[n][7]:9.2e}"
        )
    ns = list(N_LIST)
    for k in range(len(ns) - 1):
        check(f"new J^0_0 error shrinks N={ns[k]} -> {ns[k + 1]} (ratio)", sols[ns[k + 1]][6] / sols[ns[k]][6], 0.7)
        check(f"old J^0_0 error shrinks N={ns[k]} -> {ns[k + 1]} (ratio)", sols[ns[k + 1]][4] / sols[ns[k]][4], 0.7)
    check("finest grid: new J^0_0 error < 2e-2", sols[ns[-1]][6], 2e-2)
    check("finest grid: new anisotropy error < 2% of its maximum", sols[ns[-1]][7] / scale_w, 2e-2)
    check("finest grid: J^0_0 error ratio new/old", sols[ns[-1]][6] / sols[ns[-1]][4], 1.0)

    # Shallow layers (tau < 1), where the line polarization is formed.
    sh = tau_ref < 1.0
    n = ns[0]
    step = (N_REF - 1) // (n - 1)
    sh_c = np.logspace(np.log10(TAU_MIN), np.log10(TAU_MAX), n) < 1.0
    print(
        f"tau < 1, N={n}: max relative J00 error old "
        f"{np.max(np.abs(sols[n][0][sh_c] / J00_ref[::step][sh_c] - 1)):.1%}, "
        f"short characteristics "
        f"{np.max(np.abs(sols[n][8][sh_c] / J00_ref[::step][sh_c] - 1)):.1%}, "
        f"new {np.max(np.abs(sols[n][2][sh_c] / J00_ref[::step][sh_c] - 1)):.1%}"
    )

    # 4. Angular quadrature: 8-node Gauss-Legendre versus 16 nodes, and Grid.build's default.
    _, J00_ref16, J20_ref16 = solve_new(N_LIST[-1], *gauss_legendre(16))
    n = N_LIST[-1]
    step = (N_REF - 1) // (n - 1)
    print(f"angles (N={n}): 8 GL vs 16 GL: J00 {err(sols[n][2], J00_ref16):.2e}, anisotropy {err(sols[n][3], J20_ref16 / J00_ref16):.2e}")
    gb = Grid.build(make_grid(tau_c, mu8, w8).atm, make_grid(tau_c, mu8, w8).line, n_mu_half=4)
    _, J00_gb, J20_gb = solve_new(n, gb.mu, gb.w_mu)
    print(f"Grid.build default angles vs 8 GL on [-1,1]: J00 {err(J00_gb, sols[n][2]):.2e}, anisotropy {err(J20_gb / J00_gb, sols[n][3]):.2e}")

    # 5. The old defaults (eps = 1e-4, 60 iterations) are not converged.
    tau_o = np.logspace(-4, 4, 50)
    mu_o, w_o = gauss_legendre(8)
    _, J00_60, _, _, _ = old_iteration(tau_o, mu_o, w_o, 1e-4, 60)
    _, J00_6k, _, _, _ = old_iteration(tau_o, mu_o, w_o, 1e-4, 6000)
    print(f"old defaults eps=1e-4: J00 after 60 its vs 6000 its: max change {err(J00_60, J00_6k):.2e} "
          f"(deepest point {J00_60[-1]:.3f} vs {J00_6k[-1]:.3f})")

    # Plot: coarsest and finest grids against the reference.
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.6))
    n0, n1 = ns[0], ns[-1]
    ax[0].semilogx(tau_ref, J00_ref, "k-", lw=1, label=f"reference (N={N_REF})")
    ax[1].semilogx(tau_ref, w_ref, "k-", lw=1, label=f"reference (N={N_REF})")
    for n, ls in ((n0, "o"), (n1, "s")):
        t = np.logspace(np.log10(TAU_MIN), np.log10(TAU_MAX), n)
        ax[0].semilogx(t, sols[n][0], ls + "--", ms=3, label=f"old scheme N={n}")
        ax[0].semilogx(t, sols[n][8], ls + "-.", ms=3, label=f"short characteristics N={n}")
        ax[0].semilogx(t, sols[n][2], ls + ":", ms=3, label=f"new propagator N={n}")
        ax[1].semilogx(t, sols[n][1], ls + "--", ms=3, label=f"old scheme N={n}")
        ax[1].semilogx(t, sols[n][9], ls + "-.", ms=3, label=f"short characteristics N={n}")
        ax[1].semilogx(t, sols[n][3], ls + ":", ms=3, label=f"new propagator N={n}")
    ax[0].set(xlabel=r"$\tau$", ylabel=r"$J^0_0$", title=fr"Mean intensity, $\epsilon$={EPS:g}")
    ax[1].set(xlabel=r"$\tau$", ylabel=r"$J^2_0/J^0_0$", title="Anisotropy")
    ax[2].loglog(ns, [sols[n][4] for n in ns], "o-", label=r"old, $J^0_0$")
    ax[2].loglog(ns, [sols[n][6] for n in ns], "s-", label=r"new, $J^0_0$")
    ax[2].loglog(ns, [sols[n][5] for n in ns], "o--", label="old, anisotropy")
    ax[2].loglog(ns, [sols[n][10] for n in ns], "^-", label="short characteristics, $J^0_0$")
    ax[2].loglog(ns, [sols[n][11] for n in ns], "^--", label="short characteristics, anisotropy")
    ax[2].loglog(ns, [sols[n][7] for n in ns], "s--", label="new, anisotropy")
    ax[2].set(xlabel="number of depth points N", ylabel="max error vs reference", title="Refinement study")
    for a_ in ax:
        a_.grid(alpha=0.3)
        a_.legend(fontsize=7)
    fig.tight_layout()
    path = os.path.join(OUT, "compare_old_anisotropy.png")
    fig.savefig(path, dpi=130)
    print("Saved", path)


if __name__ == "__main__":
    main()
