"""Validate polarized transfer through a homogeneous slab.

Run from this directory with: python validate_homogeneous_slab.py
"""
import numpy as np
from scipy.integrate import solve_ivp

from formal_solver import propagate_constant_slab, propagation_matrix


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

print("All homogeneous-slab checks passed.")
