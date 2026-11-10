"""Regression checks for the atmosphere, line state and radiation-state interfaces."""
import numpy as np

from atmosphere import Atmosphere1D, C_LIGHT, Grid, LineModel, LineState
from grid_opacity import (fill_grid, point_emission_and_propagation,
                          point_emission_and_propagation_unresolved)
from grid_tensors import (absorption_profile, jkq_from_stokes, solve_rho,
                          solve_rho_point, update_jkq_from_stokes)


def check(name, actual, expected, rtol=1e-12, atol=1e-13):
    actual, expected = np.asarray(actual), np.asarray(expected)
    error = np.max(np.abs(actual - expected))
    np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol, err_msg=name)
    print(f"[PASS] {name}: maximum absolute difference {error:.3e}")


def rejects(name, action):
    try:
        action()
    except ValueError as exc:
        print(f"[PASS] {name}: {exc}")
    else:
        raise AssertionError(f"{name}: invalid input was accepted")


def check_scaled(name, actual, expected, tolerance=1e-12):
    scale = np.max(np.abs(expected))
    if not np.isfinite(scale) or scale == 0:
        raise AssertionError(f"{name}: reference has no finite nonzero scale")
    error = np.max(np.abs(actual - expected)) / scale
    if not np.isfinite(error) or error >= tolerance:
        raise AssertionError(f"{name}: relative maximum error {error:.3e} >= {tolerance:.1e}")
    print(f"[PASS] {name}: relative maximum error {error:.3e}")


n = 2
atm = Atmosphere1D(
    z=np.array([1.0, 0.0]), T=np.array([5000., 6500.]),
    v_z=np.array([1e5, -2e5]), v_x=np.array([3e5, -1e5]), v_y=np.array([-2e5, 4e5]),
    v_turb=np.array([1e5, 2e5]),
    B=np.array([[3., 4., 12.], [0., 0., 0.]]),
    rho=np.array([1e-7, 2e-7]), p=np.array([10., 20.]), chi_c=np.array([1e-8, 2e-8]),
)
line = LineModel(nu0=6.5e14, A_ul=2e8, g_u=1., g_l=1., J_u=1., J_l=0., mass_amu=87.6)
state = LineState(atm, line, np.array([1e8, 2e8]), np.array([1e7, 2e7]),
                  np.array([3e7, 4e7]), a=np.array([0.05, 0.1]))
grid = Grid.build(atm, line, state, n_mu_half=2, n_phi=4, n_nu=101).allocate()

strength, theta, chi = atm.magnetic_angles()
check("Cartesian field magnitude", strength, [13., 0.])
check("Cartesian field inclination", theta, [np.arctan2(5., 12.), 0.])
check("Cartesian field azimuth", chi, [np.arctan2(4., 3.), 0.])
check("mass density preserved", atm.point(0)["rho"], 1e-7)
check("line wavelength", line.lambda0, C_LIGHT / line.nu0)
check("supplied damping", state.damping, [0.05, 0.1])
check("depth-dependent destruction probability", state.destruction_probability(),
      state.C_ul * (-np.expm1(-6.62607015e-27 * line.nu0 / (1.380649e-16 * atm.T)))
      / (line.A_ul + state.C_ul * (-np.expm1(-6.62607015e-27 * line.nu0 / (1.380649e-16 * atm.T))))
)
check("eps property", state.eps, state.destruction_probability())

old_width = state.delta_nu_D.copy()
atm.T[0] *= 1.1
check("Doppler width recomputed after atmosphere change", state.delta_nu_D,
      line.doppler_width(atm.T, atm.v_turb))
if state.delta_nu_D[0] <= old_width[0]:
    raise AssertionError("Doppler width did not increase with temperature")
natural = LineState(atm, line, state.n_l, state.C_ul, state.D2)
check("natural damping fallback", natural.damping, line.A_ul / (4 * np.pi * natural.delta_nu_D))

rejects("negative D2", lambda: LineState(atm, line, state.n_l, state.C_ul, [-1., 0.]))
rejects("wrong B shape", lambda: Atmosphere1D([1., 0.], [5000., 5000.], [0., 0.], [1., 2.]))
rejects("wrong tensor shape", lambda: grid.set_JKQ(np.zeros((3, 5))))
other_atm = Atmosphere1D([1., 0.], [5000., 5000.], [0., 0.], np.zeros((2, 3)))
rejects("line state from another atmosphere", lambda: Grid.build(other_atm, line, state))

prescribed = np.zeros((n, 3, 5), complex)
prescribed[:, 0, 2], prescribed[:, 2, 2] = [1., 2.], [-0.1, 0.2]
grid.set_JKQ(prescribed)
check("prescribed tensor stored", grid.JKQ, prescribed)
fill_grid(grid)
rho_matrix = solve_rho(grid)
for k in range(n):
    check(f"rho_matrix at depth {k}", rho_matrix[k],
          solve_rho_point(prescribed[k], line, strength[k], theta[k], chi[k],
                          state.D2[k], state.C_ul[k]))

# Independently compare every ray of the grid with the legacy local geometry.
for k in range(n):
    for im, mu in enumerate(grid.mu):
        for ip, phi in enumerate(grid.phi):
            projected = (atm.v_z[k] * mu + np.sqrt(1 - mu**2)
                         * (atm.v_x[k] * np.cos(phi) + atm.v_y[k] * np.sin(phi)))
            emis, eta = point_emission_and_propagation(
                prescribed[k], line, np.array([mu]), np.array([phi]), grid.nu,
                atm.T[k], state.n_l[k], strength[k], theta[k], chi[k],
                projected / mu, atm.v_turb[k], state.D2[k], state.C_ul[k],
                chi_c=atm.chi_c[k], damping=state.damping[k],
            )
            check_scaled(f"projected-velocity emission depth {k}, ray {im},{ip}",
                         grid.emis[k, :, :, im, ip], emis[:, :, 0, 0])
            check_scaled(f"projected-velocity propagation depth {k}, ray {im},{ip}",
                         grid.eta[k, :, :, im, ip], eta[:, :, 0, 0])

profile = absorption_profile(grid)
check("profile normalized per depth and direction",
      np.einsum("knmp,n->kmp", profile, grid.w_nu), np.ones((n, len(grid.mu), len(grid.phi))))
if np.max(np.abs(profile[0, :, 0, 0] - profile[0, :, 0, 2])) == 0:
    raise AssertionError("horizontal velocity did not change opposite-azimuth profiles")
state.a[1] = 0.3
broadened = absorption_profile(grid)
change = np.max(np.abs(broadened[1] - profile[1])) / np.max(np.abs(profile[1]))
if change < 0.05:
    raise AssertionError(f"supplied damping did not affect tensor profile enough: {change:.3e}")
print(f"[PASS] supplied damping changes tensor profile: relative change {change:.3e}")
fill_grid(grid)
line_opacity = grid.eta[1, 0] - atm.chi_c[1]
opacity_norm = np.einsum("nmp,n->mp", line_opacity, grid.w_nu)
check_scaled("custom damping and horizontal velocities match local opacity profile",
             broadened[1], line_opacity / opacity_norm[None, :, :])

atm.B_unres[0] = 10.
fill_grid(grid, n_theta=2, n_chi=2)
emis, eta = point_emission_and_propagation_unresolved(
    prescribed[0], line, grid.mu, grid.phi, grid.nu, atm.T[0], state.n_l[0],
    strength[0], theta[0], chi[0], atm.B_unres[0], atm.v_z[0], atm.v_turb[0],
    state.D2[0], state.C_ul[0], n_theta=2, n_chi=2, chi_c=atm.chi_c[0],
    damping=state.damping[0], v_x=atm.v_x[0], v_y=atm.v_y[0],
)
check_scaled("unresolved field receives line state and velocities (emission)", grid.emis[0], emis)
check_scaled("unresolved field receives line state and velocities (propagation)", grid.eta[0], eta)

stokes = np.zeros_like(grid.stokes)
stokes[:, 0] = 2.
integrated = update_jkq_from_stokes(grid, stokes)
check("Stokes integration stored on grid", grid.JKQ, integrated)
check("isotropic unpolarized J00", grid.JKQ[:, 0, 2], [2., 2.])
check("isotropic unpolarized J20", grid.JKQ[:, 2, 2], [0., 0.])
check("integration matches non-mutating API", grid.JKQ, jkq_from_stokes(grid, stokes))
rejects("wrong Stokes shape", lambda: update_jkq_from_stokes(grid, stokes[..., 0]))

stokes[:, 0] = (2. + 0.3 * (grid.nu / np.max(grid.nu))[None, :, None, None]
                + 0.2 * np.cos(grid.phi)[None, None, None, :])
calculated = update_jkq_from_stokes(grid, stokes)
profile = absorption_profile(grid)
expected_j00 = np.zeros(n)
for k in range(n):
    for im in range(len(grid.mu)):
        for ip in range(len(grid.phi)):
            expected_j00[k] += (np.sum(stokes[k, 0, :, im, ip] * profile[k, :, im, ip] * grid.w_nu)
                                * grid.w_mu[im] * grid.w_phi[ip] / (4 * np.pi))
check("direction-dependent J00 matches explicit quadrature", calculated[:, 0, 2], expected_j00)
stokes[:, 1] = 0.1
stokes[:, 2] = 0.05
rotated = stokes.copy()
rotated[:, 1:3] *= -1
check("radiation tensor invariant under Stokes reference rotation",
      jkq_from_stokes(grid, stokes, gamma=0.0),
      update_jkq_from_stokes(grid, rotated, gamma=np.pi / 2))
rejects("non-finite custom profile",
        lambda: update_jkq_from_stokes(grid, stokes, profile=np.full((n, len(grid.nu)), np.nan)))

print("All model-layer checks passed.")
