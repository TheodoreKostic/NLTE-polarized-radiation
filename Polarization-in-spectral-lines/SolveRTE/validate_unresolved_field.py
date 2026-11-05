"""Validate the mean + unresolved isotropic magnetic field average.

The field at each depth is B_mean + B_unres * n, with n uniform on the sphere.
Run from this directory with: python validate_unresolved_field.py
"""
import numpy as np

from validate_collisional_depolarization import (
    LINE, MU, PHI, NU, VMIC, JKQ, J2, check, rel)
from grid_opacity import (
    point_emission_and_propagation, point_emission_and_propagation_unresolved,
    isotropic_field_directions, total_field_samples, rho2_magnetic)
from Old.functions_prt import wigner_D2

ARGS = (LINE, MU, PHI, NU, 1e-12, 1.0)
G = np.pi / 2


def unres(B, th, ch, Bu, D2=0.0, **kw):
    return point_emission_and_propagation_unresolved(
        JKQ, *ARGS, B, th, ch, Bu, 0.0, VMIC, D2, 0.0, gamma=G, **kw)


def single(B, th, ch, D2=0.0):
    return point_emission_and_propagation(
        JKQ, *ARGS, B, th, ch, 0.0, VMIC, D2, 0.0, gamma=G)


# 1. Quadrature: weights sum to one and <D_{Q0} D*_{Q'0}> = delta/5.
th, ch, wt = isotropic_field_directions(8, 8)
check("quadrature weights sum to 1", abs(wt.sum() - 1.0), 1e-14)
M = sum(w * np.outer(wigner_D2(c, t, 0.0)[:, 2], wigner_D2(c, t, 0.0)[:, 2].conj())
        for t, c, w in zip(th, ch, wt))
check("<D_{Q0} D*_{Q'0}> = delta/5", np.abs(M - np.eye(5) / 5).max(), 1e-14)

# 2. Field samples: |B_tot|^2 averages to B^2 + Bu^2; mean of the vectors is B_mean.
b, t, c, w = total_field_samples(30.0, 1.0, 0.5, 20.0)
check("<|B_tot|^2> = B^2 + Bu^2", abs(np.sum(w * b**2) - (30.0**2 + 20.0**2)) / 1300.0, 1e-13)
vec = np.stack([b * np.sin(t) * np.cos(c), b * np.sin(t) * np.sin(c), b * np.cos(t)], -1)
mean = (w[:, None] * vec).sum(0)
expect = 30.0 * np.array([np.sin(1.0) * np.cos(0.5), np.sin(1.0) * np.sin(0.5), np.cos(1.0)])
check("<B_tot> = B_mean", np.abs(mean - expect).max() / 30.0, 1e-13)

# 3. B_unres = 0 reproduces the single-field result exactly.
e0, h0 = single(50.0, 1.1, 0.7)
e1, h1 = unres(50.0, 1.1, 0.7, 0.0, n_theta=2, n_chi=2)
check("B_unres=0: emis", rel(e1[:, :, 0, 0], e0[:, :, 0, 0]), 1e-12)
check("B_unres=0: eta", rel(h1[:, :, 0, 0], h0[:, :, 0, 0]), 1e-12)

# 4. A small unresolved component barely changes the mean-field result.
e2, h2 = unres(50.0, 1.1, 0.7, 1e-6)
check("B_unres->0 is continuous (emis)", rel(e2[:, :, 0, 0], e0[:, :, 0, 0]), 1e-5)
check("B_unres->0 is continuous (eta)", rel(h2[:, :, 0, 0], h0[:, :, 0, 0]), 1e-5)

# 5. B_mean = 0: purely isotropic field; no net V or Faraday rotation survives.
e3, h3 = unres(0.0, 0.0, 0.0, 50.0)
for name, k in (("eta_V", 3), ("rho_V", 6)):
    check(f"isotropic field: averaged {name} vanishes",
          np.abs(h3[k]).max() / np.abs(h3[0]).max(), 1e-12)

# 6. Strong isotropic field: <rho^2> -> J^2 / 5 (rank-2 depolarization).
th, ch, wt = isotropic_field_directions(8, 8)
avg = sum(w * wigner_D2(c, t, 0.0) @ rho2_magnetic(J2, LINE, 1e5, t, c, 0.0, 0.0)
          for t, c, w in zip(th, ch, wt))
check("strong isotropic field: <rho^2> -> J^2/5", np.abs(avg - J2 / 5).max(), 1e-3)

# 7. Direction quadrature converges. Smooth case: unresolved part larger than the mean.
eA, hA = unres(5.0, 1.1, 0.7, 50.0, n_theta=16, n_chi=16)
eB, hB = unres(5.0, 1.1, 0.7, 50.0, n_theta=24, n_chi=24)
check("quadrature convergence, B_unres > B (emis)", rel(eA, eB), 1e-10)
check("quadrature convergence, B_unres > B (eta)", rel(hA, hB), 1e-10)
# Hard case: B ~ B_unres with Hu >> 1, where |B_tot| can pass near zero. Compare each
# quadrature with a 48x48 reference; every doubling of nodes must at least halve the error.
ref = unres(50.0, 1.1, 0.7, 50.0, n_theta=48, n_chi=48)[0]
err = {n: rel(unres(50.0, 1.1, 0.7, 50.0, n_theta=n, n_chi=n)[0], ref) for n in (8, 16, 32)}
print("  hard-case quadrature error vs 48x48:", {n: f"{e:.1e}" for n, e in err.items()})
check("hard case: error(16)/error(8) < 0.5", err[16] / err[8], 0.5)
check("hard case: error(32)/error(16) < 0.5", err[32] / err[16], 0.5)
check("hard case: 32x32 accurate to 1e-2", err[32], 1e-2)

# 8. Collisions reduce the linearly polarized emission (same quadrature throughout).
lin = lambda e: np.sqrt(e[1] ** 2 + e[2] ** 2).max()
ratios = [0.0, 0.5, 1.0, 3.0, 10.0, 100.0]
amps = [lin(unres(5.0, 1.1, 0.7, 5.0, D2=r * LINE.A_ul, n_theta=12, n_chi=12)[0]) for r in ratios]
print("  linear amplitude vs D2/A:", {r: f"{a:.3e}" for r, a in zip(ratios, amps)})
check("amplitude decreases monotonically with D2",
      float(np.max(np.diff(amps))), 0.0)
# For D2 >> A and Hu, rank-2 relaxation is 1 + D2/A, so amp * (1 + D2/A) tends to the
# unmagnetized, collision-free amplitude.
big = 1e4
amp_big = lin(unres(5.0, 1.1, 0.7, 5.0, D2=big * LINE.A_ul, n_theta=12, n_chi=12)[0]) * (1 + big)
amp_b0 = lin(unres(0.0, 0.0, 0.0, 0.0, n_theta=2, n_chi=2)[0])
check("strong D2: amp*(1+D2/A) -> unmagnetized amplitude", abs(amp_big / amp_b0 - 1.0), 1e-3)
print("All unresolved-field checks passed.")
