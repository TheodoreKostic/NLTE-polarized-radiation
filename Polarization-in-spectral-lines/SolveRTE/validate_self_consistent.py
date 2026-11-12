"""Validate short characteristics and the self-consistent Lambda iteration.

Run from this directory with: python validate_self_consistent.py
"""
import warnings

import numpy as np

from atmosphere import Atmosphere1D, Grid, LineModel, LineState
from formal_solver import solve_short_characteristics, solve_unpolarized_transfer
from grid_opacity import fill_grid, planck
from grid_tensors import absorption_profile, jkq_from_stokes, tensor_basis

NU0, A_UL, T = 6.5e14, 2e8, 6000.0
LINE = LineModel(NU0, A_UL, 1.0, 1.0, 1, 0, 87.6)


def report(name, passed, detail=""):
    print(f"[{'PASS' if passed else 'FAIL'}] {name} {detail}".rstrip())
    if not passed:
        raise AssertionError(f"{name} {detail}")


def check(name, actual, expected, tol):
    actual, expected = np.asarray(actual), np.asarray(expected)
    error = np.max(np.abs(actual - expected)) / max(np.max(np.abs(expected)), 1e-300)
    report(name, np.isfinite(error) and error < tol, f"relative error {error:.2e} (tol {tol:.0e})")


def rejects(name, action):
    try:
        action()
    except ValueError as exc:
        report(name, True, f"({exc})")
    else:
        report(name, False, "invalid input was accepted")


# 1. Short characteristics reproduces the exact solution for a source linear in tau.
rng = np.random.default_rng(3)
ndepth, nnu, nmu, nphi = 12, 3, 4, 2
tau = np.cumsum(rng.uniform(0.02, 2.0, (ndepth, nnu)), axis=0) - 0.02
mu = np.array([-0.9, -0.3, 0.3, 0.9])
intercept = rng.uniform(0.5, 1.5, (3, nnu, nmu, nphi))
slope = rng.uniform(-0.1, 0.1, (3, nnu, nmu, nphi))
source = intercept[None] + slope[None] * tau[:, None, :, None, None]
top, bottom = rng.uniform(0, 1, (2, 3, nnu, nmu, nphi))
check("short characteristics equals exact transfer for a linear source",
      solve_short_characteristics(source, tau, mu, top, bottom),
      solve_unpolarized_transfer(source, tau, mu, top, bottom), 1e-10)
rejects("short characteristics needs two depth points",
        lambda: solve_short_characteristics(source[:1], tau[:1], mu, top, bottom))
flat_tau = tau.copy()
flat_tau[5] = flat_tau[4]
rejects("short characteristics needs strictly increasing tau",
        lambda: solve_short_characteristics(source, flat_tau, mu, top, bottom))
rejects("short characteristics validates boundaries",
        lambda: solve_short_characteristics(source, tau, mu, top[:, :, :-1], bottom))


# 2. Self-consistent solution of a homogeneous, unmagnetized slab.
def make_grid(C_ul, tau_line=20.0, n=30):
    """Slab whose fixed opacity is the line opacity eta_I(nu), with tau_line at line centre."""
    def build(n_l, z):
        atm = Atmosphere1D(z, np.full(n, T), np.zeros(n), np.zeros((n, 3)))
        state = LineState(atm, LINE, np.full(n, n_l), np.full(n, C_ul), np.zeros(n))
        grid = Grid.build(atm, LINE, state, n_mu_half=3, n_phi=4, n_nu=41).allocate()
        grid.JKQ[:, 0, 2] = 1.0
        fill_grid(grid)
        return grid

    trial = build(1.0, np.linspace(1.0, 0.0, n))
    n_l = 1.0 / trial.eta[0, 0].max()
    depth = np.concatenate([[0.0], np.logspace(-3, np.log10(tau_line), n - 1)])
    grid = build(n_l, depth[-1] - depth)
    return grid, grid.eta[0, 0, :, 0, 0].copy()


B_NU = planck(NU0, T)

# eps -> 1: the source is the Planck function everywhere, so the solution is analytic.
grid, chi = make_grid(C_ul=1e20)
grid.stokes[:, 3] = 7.0
result = grid.solve_rte_sc(chi, tol=1e-10)
report("thermal limit converges", result.converged, f"in {result.iterations} iterations")
expected = np.zeros_like(grid.stokes[:, :3])
for m, ray_mu in enumerate(grid.mu):
    expected[:, 0, :, m, :] = B_NU if ray_mu > 0 else (
        B_NU * -np.expm1(-grid.tau / abs(ray_mu)))[:, :, None]
check("thermal limit: I from the analytic constant-source solution",
      grid.stokes[:, 0], expected[:, 0], 1e-8)
report("thermal limit: Q and U vanish", np.max(np.abs(grid.stokes[:, 1:3])) < 1e-9 * B_NU)
check("grid.stokes V is untouched", grid.stokes[:, 3], np.full_like(grid.stokes[:, 3], 7.0), 1e-300)
check("grid.tau is stored", grid.tau, grid.calc_tau(chi), 1e-300)

# Scattering-dominated slab (eps ~ 0.1): a genuinely self-consistent solution.
grid, chi = make_grid(C_ul=2.2e7)
eps = grid.line_state.eps[0]
result = grid.solve_rte_sc(chi, tol=1e-9, max_iter=2000)
report("scattering slab converges", result.converged,
       f"eps = {eps:.3f}, {result.iterations} iterations")
report("relative change decreases to tolerance",
       result.history[-1] < 1e-9 < result.history[0] and result.history[-1] < result.history[-5])

# The converged tensor and Stokes field are a fixed point of the iteration.
j = LINE.calc_j(grid)
stokes = np.zeros_like(grid.stokes)
top_default, bottom_default = grid._default_boundaries()
stokes[:, :3] = solve_short_characteristics(
    LINE.calc_S(j, chi), grid.tau, grid.mu, top_default, bottom_default)
updated = jkq_from_stokes(grid, stokes, basis=tensor_basis(grid.mu, grid.phi),
                          profile=absorption_profile(grid))
check("fixed point: J^K_Q reproduced from its own emissivity",
      updated, grid.JKQ, 1e-7)
check("fixed point: Stokes field reproduced", stokes[:, :3], grid.stokes[:, :3], 1e-7)

J00, J20 = grid.JKQ[:, 0, 2].real, grid.JKQ[:, 2, 2].real
report("J^0_0 is below the Planck function and increases with depth",
       np.all(J00 < B_NU) and np.all(np.diff(J00) > 0))
report("J^0_0 thermalizes at depth", J00[-1] > 0.95 * B_NU, f"({J00[-1] / B_NU:.4f} B)")
report("J^2_0 is positive at the surface and weaker at depth than at the surface",
       J20[0] > 0 and abs(J20[-1] / J00[-1]) < abs(J20[0] / J00[0]),
       f"(J^2_0/J^0_0 = {J20[0] / J00[0]:.3f} at the top, {J20[-1] / J00[-1]:.4f} at the bottom)")
report("J^1_Q vanishes without circular polarization", np.max(np.abs(grid.JKQ[:, 1])) == 0)
q_limb = grid.stokes[0, 1, :, 3, 0]  # most grazing outgoing ray at the top
report("scattering polarization appears at the surface",
       np.max(np.abs(q_limb)) > 1e-3 * np.max(grid.stokes[0, 0, :, 3, 0]))
report("azimuthal symmetry: U vanishes and Q is independent of phi",
       np.max(np.abs(grid.stokes[:, 2])) < 1e-9 * B_NU
       and np.ptp(grid.stokes[:, 1], axis=-1).max() < 1e-9 * B_NU)


def lambda_iteration_fs(grid, chi, tol=1e-7, max_iter=3000):
    """Reference loop using the exact linear-source propagator instead of short characteristics."""
    tau = grid.calc_tau(chi)
    top, bottom = grid._default_boundaries()
    basis, profile = tensor_basis(grid.mu, grid.phi), absorption_profile(grid)
    start = np.zeros((grid.atm.N, 3, 5), complex)
    start[:, 0, 2] = planck(NU0, grid.atm.T)
    grid.set_JKQ(start)
    stokes = np.zeros_like(grid.stokes)
    for _ in range(max_iter):
        source = LINE.calc_S(LINE.calc_j(grid), chi)
        stokes[:, :3] = solve_unpolarized_transfer(source, tau, grid.mu, top, bottom)
        updated = jkq_from_stokes(grid, stokes, basis=basis, profile=profile)
        change = np.max(np.abs(updated - grid.JKQ) / np.abs(updated[:, 0, 2])[:, None, None])
        grid.set_JKQ(updated)
        if change < tol:
            break
    grid.stokes[:, :3] = stokes[:, :3]


# The two formal solvers solve the same discrete problem up to interpolation error,
# so their converged solutions must approach each other as the depth grid is refined.
differences = []
for n_depth in (30, 60):
    sc_grid, chi_n = make_grid(C_ul=2.2e7, n=n_depth)
    fs_grid, _ = make_grid(C_ul=2.2e7, n=n_depth)
    sc_grid.solve_rte_sc(chi_n, tol=1e-7, max_iter=3000)
    lambda_iteration_fs(fs_grid, chi_n)
    differences.append(np.max(np.abs(sc_grid.stokes[:, :3] - fs_grid.stokes[:, :3])) / B_NU)
report("short-characteristics and exact-propagator iterations agree",
       differences[0] < 3e-2, f"(difference {differences[0]:.2e} of B at 30 depths)")
report("their difference shrinks under depth refinement", differences[1] < differences[0] / 2,
       f"({differences[0]:.2e} -> {differences[1]:.2e})")

# Restarting from the converged tensor stays converged; non-convergence is reported, not hidden.
restart = grid.solve_rte_sc(chi, JKQ0=grid.JKQ.copy(), tol=1e-7)
report("restart from converged tensor converges immediately",
       restart.converged and restart.iterations == 1, f"({restart.iterations} iteration)")
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    short = grid.solve_rte_sc(chi, tol=1e-12, max_iter=2)
report("non-convergence is flagged",
       not short.converged and short.iterations == 2
       and any(issubclass(w.category, RuntimeWarning) for w in caught))

rejects("tol must be positive", lambda: grid.solve_rte_sc(chi, tol=0.0))
rejects("max_iter must be positive", lambda: grid.solve_rte_sc(chi, max_iter=0))
rejects("opacity must be positive", lambda: grid.solve_rte_sc(-chi))
rejects("starting tensor shape is checked", lambda: grid.solve_rte_sc(chi, JKQ0=np.zeros((3, 5))))
rejects("boundary shape is checked",
        lambda: grid.solve_rte_sc(chi, top=np.zeros((3, 1, 1, 1))))
print("All self-consistent transfer checks passed.")
