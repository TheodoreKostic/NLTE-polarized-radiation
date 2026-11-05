"""Regression checks for Voigt-weighted radiation tensors.

Run from this directory with: python validate_radiation_tensor.py
"""
import numpy as np

from atmosphere import Atmosphere1D, C_LIGHT, Grid, LineModel
from grid_opacity import eta0, phi00_norm, point_emission_and_propagation, stimulated_factor
from grid_tensors import absorption_profile, jkq_from_stokes, tensor_basis
from line_profiles import voigt_function


def check(name, error, tolerance):
    passed = np.isfinite(error) and error < tolerance
    print(f"[{'PASS' if passed else 'FAIL'}] {name}: {error:.2e} (tol {tolerance:.0e})")
    if not passed:
        raise AssertionError(
            f"{name}: error {error:.3e} is not below tolerance {tolerance:.3e}"
        )


line = LineModel(
    4.3e14, 5e7, 1.0, 1.0, 1.0, 0.0, 40.0,
    n_l=np.array([1e10, 2e10]), C_ul=np.zeros(2),
)
atmosphere = Atmosphere1D(
    z=np.array([1.0, 0.0]),
    T=np.array([5000.0, 6500.0]),
    B=np.zeros(2),
    theta_B=np.zeros(2),
    chi_B=np.zeros(2),
    v_los=np.array([0.0, 4e5]),
    D2=np.zeros(2),
    vmic=np.array([1e5, 2e5]),
)
grid = Grid.build(atmosphere, line, n_mu_half=2, n_phi=4, n_nu=401, nu_max_doppler=8.0)
profile = absorption_profile(grid)

x_test = np.linspace(-6.0, 6.0, 101)
gaussian = np.exp(-x_test**2) / np.sqrt(np.pi)
check(
    "Voigt profile has Gaussian zero-damping limit",
    np.max(np.abs(voigt_function(x_test, 0.0).real - gaussian)),
    1e-14,
)

profile_norm = np.einsum("knm,n->km", profile, grid.w_nu)
check("profile normalized at every depth and ray", np.max(np.abs(profile_norm - 1.0)), 1e-13)

zero_j = np.zeros((3, 5), dtype=complex)
for k in range(atmosphere.N):
    dnu = line.doppler_width(atmosphere.T[k], atmosphere.vmic[k])
    a = line.A_ul / (4.0 * np.pi * dnu)
    _, eta = point_emission_and_propagation(
        zero_j,
        line,
        grid.mu,
        grid.phi,
        grid.nu,
        atmosphere.T[k],
        line.n_l[k],
        0.0,
        0.0,
        0.0,
        atmosphere.v_los[k],
        atmosphere.vmic[k],
        0.0,
        0.0,
    )
    x = (
        -grid.nu[:, None]
        + line.nu0 * atmosphere.v_los[k] * grid.mu[None, :] / C_LIGHT
    ) / dnu
    raw_profile = voigt_function(x, a).real
    amplitude = stimulated_factor(line.nu0, atmosphere.T[k]) * eta0(line, line.n_l[k]) / (phi00_norm() * dnu)
    opacity_profile = eta[0, :, :, 0] / amplitude
    check(
        f"local opacity uses shared Voigt profile at depth {k}",
        np.max(np.abs(opacity_profile - raw_profile)),
        2e-13,
    )
    normalized = raw_profile / np.einsum("nm,n->m", raw_profile, grid.w_nu)[None, :]
    check(
        f"tensor profile matches local opacity at depth {k}",
        np.max(np.abs(profile[k] - normalized)),
        2e-13,
    )

stokes = np.zeros((atmosphere.N, 4, len(grid.nu), len(grid.mu), len(grid.phi)))
nu_scaled = grid.nu[None, :, None, None] / grid.nu.max()
mu = grid.mu[None, None, :, None]
phi = grid.phi[None, None, None, :]
stokes[:, 0] = 1.0 + 0.1 * nu_scaled + 0.05 * mu + 0.02 * np.cos(phi)
stokes[:, 1] = 0.03 * np.sin(phi) + 0.01 * nu_scaled
stokes[:, 2] = 0.02 * mu * np.cos(phi)
stokes[:, 3] = 0.01 * np.sin(phi)

basis = tensor_basis(grid.mu, grid.phi)
computed = jkq_from_stokes(grid, stokes, basis=basis, profile=profile)
direct = np.zeros_like(computed)
for k in range(atmosphere.N):
    for K in range(3):
        for q_index in range(5):
            total = 0.0j
            for i in range(4):
                for n in range(len(grid.nu)):
                    for m in range(len(grid.mu)):
                        for p in range(len(grid.phi)):
                            total += (
                                basis[i, K, q_index, m, p]
                                * stokes[k, i, n, m, p]
                                * profile[k, n, m]
                                * grid.w_nu[n]
                                * grid.w_mu[m]
                                * grid.w_phi[p]
                            )
            direct[k, K, q_index] = total / (4.0 * np.pi)
error = np.max(np.abs(computed - direct)) / np.max(np.abs(direct))
check("tensor contraction matches direct weighted integral", error, 2e-13)

print("All radiation-tensor checks passed.")
