"""Validate polarized transfer through a homogeneous slab.

Run from this directory with: python validate_homogeneous_slab.py
"""
import numpy as np
from scipy.integrate import quad, solve_ivp

from atmosphere import Atmosphere1D, Grid, LineModel, LineState
from formal_solver import propagate_constant_slab, propagation_matrix, solve_unpolarized_transfer
from grid_opacity import planck


def check(name, actual, expected, tolerance=1e-11):
    error = np.max(np.abs(actual - expected))
    scale = max(1.0, np.max(np.abs(expected)))
    relative_error = error / scale
    passed = np.isfinite(relative_error) and relative_error < tolerance
    print(
        f"[{'PASS' if passed else 'FAIL'}] {name}: "
        f"{relative_error:.2e} (tol {tolerance:.0e})"
    )
    if not passed:
        raise AssertionError(
            f"{name}: relative error {relative_error:.3e} "
            f"is not below tolerance {tolerance:.3e}"
        )


incident = np.array([1.2, 0.1, -0.05, 0.02])
distance = 2.3

# Vacuum and pure absorption have independent closed-form solutions.
zero_coefficients = np.zeros(7)
zero_emissivity = np.zeros(4)
vacuum = propagate_constant_slab(
    zero_emissivity, zero_coefficients, distance, incident
)
check("zero coefficients preserve incident Stokes", vacuum, incident)

eta_i = 0.7
absorption = np.array([eta_i, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
absorbed = propagate_constant_slab(
    zero_emissivity, absorption, distance, incident
)
check(
    "pure absorption follows Beer-Lambert law",
    absorbed,
    incident * np.exp(-eta_i * distance),
)

# Scalar source function: S_out = S_source + (S_in-S_source) exp(-eta L).
source = np.array([0.8, 0.0, 0.0, 0.0])
scalar_emissivity = eta_i * source
scalar = propagate_constant_slab(
    scalar_emissivity, absorption, distance, incident
)
scalar_expected = source + (incident - source) * np.exp(-eta_i * distance)
check("constant scalar source solution", scalar, scalar_expected)

# Compare coupled polarized transfer with an independent adaptive ODE solve.
coefficients = np.array([0.9, 0.12, -0.08, 0.05, 0.07, -0.04, 0.11])
emissivity = np.array([0.65, 0.04, -0.03, 0.02])
matrix = propagation_matrix(coefficients)


def transfer_ode(_s, stokes):
    return emissivity - matrix @ stokes


reference = solve_ivp(
    transfer_ode,
    (0.0, distance),
    incident,
    method="DOP853",
    rtol=2e-13,
    atol=2e-14,
).y[:, -1]
coupled = propagate_constant_slab(emissivity, coefficients, distance, incident)
check("coupled IQUV transfer agrees with independent ODE integration", coupled, reference)

first_half = propagate_constant_slab(emissivity, coefficients, distance / 2, incident)
two_halves = propagate_constant_slab(
    emissivity, coefficients, distance / 2, first_half
)
check("homogeneous propagation composes across slab subdivisions", two_halves, coupled)

atm = Atmosphere1D([4., 3.8, 1., 0.], [5000., 5200., 5500., 6000.],
                   [0.] * 4, np.zeros((4, 3)))
line = LineModel(6.5e14, 2e8, 1., 1., 1., 0., 40.)
line_state = LineState(atm, line, [1.] * 4, [0.] * 4, [0.] * 4)
grid = Grid.build(atm, line, line_state, n_mu_half=2, n_phi=2, n_nu=3).allocate()
tau = grid.calc_tau([0.05, 0.5, 2.])
shape = (3, len(grid.nu), len(grid.mu), len(grid.phi))
intercept = np.arange(np.prod(shape)).reshape(shape) / 30 + 0.2
slope = np.broadcast_to(np.array([0.2, -0.1, 0.05])[:, None, None, None], shape)
source = intercept[None] + slope[None] * tau[:, None, :, None, None]
top = np.full(shape, 0.7)
bottom = np.full(shape, -0.3)
grid.stokes[:, 3] = 9.
actual = grid.solve_rte_fs(source, tau, top=top, bottom=bottom)
expected = np.empty_like(actual)
for m, ray_mu in enumerate(grid.mu):
    boundary_depth = -1 if ray_mu > 0 else 0
    boundary = bottom if ray_mu > 0 else top
    for k in range(atm.N):
        delta = np.abs(tau[k] - tau[boundary_depth]) / abs(ray_mu)
        particular = source[k, :, :, m, :] + ray_mu * slope[:, :, m, :]
        boundary_particular = (source[boundary_depth, :, :, m, :]
                               + ray_mu * slope[:, :, m, :])
        expected[k, :, :, m, :] = particular + (
            boundary[:, :, m, :] - boundary_particular
        ) * np.exp(-delta)[None, :, None]
check("linear source and prescribed boundaries for both ray directions", actual, expected)
check("grid stores I/Q/U solution", grid.stokes[:, :3], actual)
check("grid leaves V untouched", grid.stokes[:, 3], np.full_like(grid.stokes[:, 3], 9.))

zero_source = np.zeros_like(source)
defaults = grid.solve_rte_fs(zero_source, tau)
for m, ray_mu in enumerate(grid.mu):
    if ray_mu < 0:
        check("dark top boundary", defaults[:, :, :, m, :], np.zeros_like(defaults[:, :, :, m, :]))
    else:
        expected_i = planck(line.nu0, atm.T[-1]) * np.exp(-(tau[-1] - tau) / ray_mu)
        np.testing.assert_allclose(
            defaults[:, 0, :, m, :],
            np.broadcast_to(expected_i[:, :, None], defaults[:, 0, :, m, :].shape),
            rtol=1e-12, atol=0)
        print("[PASS] thermal bottom boundary")
        check("unpolarized bottom boundary", defaults[:, 1:, :, m, :],
              np.zeros_like(defaults[:, 1:, :, m, :]))

# For a source increasing linearly from zero to one, check thin and thick cells.
for dt in (1e-12, 1e-3, 1e-2, 1e6):
    cell_source = np.zeros((2, 3, 1, 1, 1))
    cell_source[1] = 1.
    cell_tau = np.array([[0.], [dt]])
    boundary = np.zeros((3, 1, 1, 1))
    result = solve_unpolarized_transfer(cell_source, cell_tau, [-1.], boundary, boundary)
    expected_value = (dt * (1/2 + dt * (-1/6 + dt * (1/24 + dt * (
        -1/120 + dt * (1/720 - dt/5040))))) if dt < 1 else 1 - 1 / dt)
    np.testing.assert_allclose(result[1], expected_value, rtol=1e-11, atol=0)
    print(f"[PASS] linear source in cell of optical thickness {dt:g}")

one_source = np.zeros((1, 3, 1, 2, 1))
one_top = np.full((3, 1, 2, 1), 2.)
one_bottom = np.full_like(one_top, 3.)
one_result = solve_unpolarized_transfer(one_source, [[0.]], [-0.5, 0.5], one_top, one_bottom)
np.testing.assert_array_equal(one_result[0, :, :, 0], one_top[:, :, 0])
np.testing.assert_array_equal(one_result[0, :, :, 1], one_bottom[:, :, 1])
print("[PASS] single-depth grid preserves each incoming boundary")

grid.JKQ[:, 0, 2] = 1.
grid.JKQ[:, 2, 2] = 0.1
emissivity = line.calc_j(grid)
physical_source = line.calc_S(emissivity, [0.05, 0.5, 2.])
zero_boundary = np.zeros(shape)
physical_solution = grid.solve_rte_fs(physical_source, tau, top=zero_boundary, bottom=zero_boundary)
m, f, p = len(grid.mu) - 1, 1, 0
ray_mu = grid.mu[m]
for i in range(3):
    profile = physical_source[:, i, f, m, p]
    integral = quad(
        lambda t: np.interp(t, tau[:, f], profile) * np.exp(-t / ray_mu) / ray_mu,
        0., tau[-1, f], points=tau[1:-1, f], epsabs=1e-30, epsrel=1e-11,
    )[0]
    reference = integral
    np.testing.assert_allclose(physical_solution[0, i, f, m, p], reference,
                               rtol=1e-11, atol=1e-25)
print("[PASS] emissivity-to-source-to-transfer agrees with direct quadrature")

for bad_tau in (np.ones((4, 2)), np.full_like(tau, np.nan), -tau, tau[::-1]):
    try:
        grid.solve_rte_fs(source, bad_tau, top=top, bottom=bottom)
    except ValueError:
        pass
    else:
        raise AssertionError("invalid optical depth was accepted")
try:
    solve_unpolarized_transfer(source, tau, np.zeros_like(grid.mu), top, bottom)
except ValueError:
    pass
else:
    raise AssertionError("horizontal rays were accepted")
bad_source = source.copy()
bad_source[0, 0, 0, 0, 0] = np.nan
for bad_s, bad_top in ((bad_source, top), (source, top[:, :, :-1])):
    try:
        grid.solve_rte_fs(bad_s, tau, top=bad_top, bottom=bottom)
    except ValueError:
        pass
    else:
        raise AssertionError("invalid source or boundary was accepted")
print("[PASS] invalid optical depths, rays, sources and boundaries rejected")

print("All homogeneous-slab checks passed.")
