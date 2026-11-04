"""Validate depolarizing collisions (D2) in the local emissivity.

Run from this directory with: python validate_collisional_depolarization.py
"""
import numpy as np

from atmosphere import LineModel, C_LIGHT
from grid_opacity import point_emission_and_propagation, rho2_magnetic
from grid_tensors import solve_rho_point
from Old.functions_prt import wigner_D2

NU0, DND = 4.3e14, 4e9
LINE = LineModel(NU0, 5e7, 1.0, 1.0, 1, 0, 40)
VMIC = DND * C_LIGHT / NU0
X = np.linspace(-6, 6, 241)
NU = -X * DND
THETA_OBS, CHI_OBS = 0.9, 0.4
MU, PHI = np.array([np.cos(THETA_OBS)]), np.array([CHI_OBS])

# J^2_Q obeying the hermiticity relation J^2_{-Q} = (-1)^Q conj(J^2_Q)
J2 = np.zeros(5, complex)
J2[2] = 0.30
J2[3], J2[1] = 0.05 + 0.04j, -(0.05 - 0.04j)
J2[4], J2[0] = 0.08 - 0.03j, 0.08 + 0.03j
JKQ = np.zeros((3, 5), complex)
JKQ[2] = J2


def check(name, error, tol):
    ok = np.isfinite(error) and error < tol
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {error:.2e} (tol {tol:.0e})")
    if not ok:
        raise AssertionError(f"{name}: error {error:.3e} not below {tol:.3e}")


def emission(B, th_B, chi_B, D2, C_ul=0.0, J00=0.0):
    J = JKQ.copy()
    J[0, 2] = J00
    eps, _ = point_emission_and_propagation(
        J, LINE, MU, PHI, NU, 1e-12, 1.0, B, th_B, chi_B, 0.0, VMIC, D2, C_ul,
        gamma=np.pi / 2)
    return eps[:, :, 0, 0]


def rel(a, b):
    return np.abs(a - b).max() / np.abs(b).max()


# 1. B = 0: the whole rank-2 term is reduced by 1 / (1 + D2/A).
for D2 in (0.5 * LINE.A_ul, 2.0 * LINE.A_ul, 10.0 * LINE.A_ul):
    ref = emission(0.0, 0.3, 0.2, 0.0) / (1.0 + D2 / LINE.A_ul)
    check(f"B=0: eps scales as 1/(1+D2/A), D2/A={D2 / LINE.A_ul:g}",
          rel(emission(0.0, 0.3, 0.2, D2), ref), 1e-12)

# 2. D2 and C_ul enter the rank-2 relaxation only through their sum.
for B, th, chi in [(0.0, 0.0, 0.0), (5.69, 1.1, 0.7), (300.0, 0.6, 2.0)]:
    ra = rho2_magnetic(J2, LINE, B, th, chi, 0.7 * LINE.A_ul, 0.0)
    rb = rho2_magnetic(J2, LINE, B, th, chi, 0.0, 0.7 * LINE.A_ul)
    check(f"rho^2 depends on D2 + C_ul only (B={B:g})", np.abs(ra - rb).max(), 1e-15)

# 3. Strong depolarization removes all rank-2 emission; Q, U -> 0 and I -> J00 term.
J00 = 1.0
strong = emission(5.69, 1.1, 0.7, 1e9 * LINE.A_ul, J00=J00)
J_only00 = np.zeros((3, 5), complex)
J_only00[0, 2] = J00
ref = point_emission_and_propagation(
    J_only00, LINE, MU, PHI, NU, 1e-12, 1.0, 5.69, 1.1, 0.7, 0.0, VMIC, 0.0, 0.0,
    gamma=np.pi / 2)[0][:, :, 0, 0]
check("D2 -> infinity: emission reduces to the J00 (unpolarized-pump) term",
      rel(strong, ref), 1e-8)

# 4. Hanle operator: rho^2 in the vertical frame equals solve_rho_point (shared Larmor constant).
D2 = 0.4 * LINE.A_ul
for B, th, chi in [(0.0, 0.5, 0.3), (5.69, 1.1, 0.7), (50.0, 2.0, 4.0)]:
    mag = rho2_magnetic(J2, LINE, B, th, chi, D2, 0.0)
    vert = wigner_D2(chi, th, 0.0) @ mag
    other = solve_rho_point(JKQ, LINE, B, th, chi, D2)[2]
    err = np.abs(vert - other).max() / np.abs(other).max()
    check(f"grid_opacity rho^2 agrees with solve_rho_point (B={B:g})", err, 1e-12)

print("All collisional-depolarization checks passed.")
