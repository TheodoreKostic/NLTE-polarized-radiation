"""Validate the photon destruction probability, stimulated emission and continuum opacity.

Run from this directory with: python validate_continuum.py
"""
import numpy as np

from atmosphere import Atmosphere1D, Grid, LineModel, C_LIGHT, K_B
from grid_opacity import (
    H_PLANCK, destruction_probability, fill_grid, planck, point_emission_and_propagation,
    point_emission_and_propagation_unresolved, stimulated_factor)

NU0, A_UL = 6.5e14, 2e8
DND = 1e9
LINE = LineModel(NU0, A_UL, 1.0, 1.0, 1, 0, 40, n_l=np.ones(1), C_ul=np.zeros(1))
VMIC = DND * C_LIGHT / NU0
NU = np.linspace(-6 * DND, 6 * DND, 121)
MU, PHI = np.array([0.7]), np.array([0.3])
T = 6000.0


def check(name, error, tol):
    ok = np.isfinite(error) and error < tol
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {error:.2e} (tol {tol:.0e})")
    if not ok:
        raise AssertionError(f"{name}: error {error:.3e} not below {tol:.3e}")


def jtensor(j00=0.0, j20=0.0):
    J = np.zeros((3, 5), complex)
    J[0, 2], J[2, 2] = j00, j20
    return J


def local(J, B=0.0, th=0.0, ch=0.0, D2=0.0, C_ul=0.0, n_l=1.0, T=T, **kw):
    emis, eta = point_emission_and_propagation(
        J, LINE, MU, PHI, NU, T, n_l, B, th, ch, 0.0, VMIC, D2, C_ul, gamma=np.pi / 2, **kw)
    return emis[:, :, 0, 0], eta[:, :, 0, 0]


rel = lambda a, b: np.abs(a - b).max() / np.abs(b).max()

# 1. Destruction probability: the formula, its limits and its depth dependence.
x = H_PLANCK * NU0 / (K_B * T)
for C in (1e6, 2e8, 5e11):
    expected = C * (1 - np.exp(-x)) / (A_UL + C * (1 - np.exp(-x)))
    check(f"eps formula, C_ul = {C:g}", abs(destruction_probability(LINE, T, C) - expected), 1e-14)
check("eps = 0 without collisions", destruction_probability(LINE, T, 0.0), 1e-300)
check("eps -> 1 for C_ul >> A_ul", 1.0 - destruction_probability(LINE, T, 1e18), 1e-9)
check("eps without stimulated emission is C/(A + C)",
      abs(destruction_probability(LINE, T, 3e8, stimulated=False) - 3e8 / (A_UL + 3e8)), 1e-15)
check("f -> 1 at low temperature", abs(stimulated_factor(NU0, 50.0) - 1.0), 1e-12)
T_depth, C_depth = np.array([4000.0, 6000.0, 9000.0]), np.array([1e7, 1e8, 1e9])
f_depth = 1 - np.exp(-H_PLANCK * NU0 / (K_B * T_depth))
check("eps at several depths (T and C_ul arrays)",
      np.abs(destruction_probability(LINE, T_depth, C_depth)
             - C_depth * f_depth / (A_UL + C_depth * f_depth)).max(), 1e-14)

# 2. Stimulated emission scales the whole line opacity by f at every frequency (B = 0).
J0 = jtensor(1.0)
_, eta_s = local(J0)
_, eta_n = local(J0, stimulated=False)
check("line opacity ratio equals f at all frequencies", rel(eta_s[0] / eta_n[0], np.full(len(NU), stimulated_factor(NU0, T))), 1e-13)

# 3. Line source function (1 - eps) J00 + eps B_nu at B = 0, isotropic J00, any C_ul.
Bnu = planck(NU0, T)
for C in (0.0, 1e7, 2e8, 1e10):
    eps = destruction_probability(LINE, T, C)
    e, h = local(J0, C_ul=C)
    S_line = e[0] / h[0]
    expected = (1 - eps) * 1.0 + eps * Bnu
    check(f"line source = (1 - eps) J00 + eps B_nu, C_ul = {C:g}", rel(S_line, np.full(len(NU), expected)), 1e-12)

# 4. Pure continuum (no line absorbers): eta_I = chi_c, j_I = chi_c B_nu, nothing polarized.
chi_c = 3e-6
e, h = local(jtensor(1.0, 0.1), B=50.0, th=1.0, ch=0.4, n_l=0.0, chi_c=chi_c)
check("pure continuum: eta_I = chi_c", rel(h[0], np.full(len(NU), chi_c)), 1e-14)
check("pure continuum: j_I = chi_c B_nu", rel(e[0], np.full(len(NU), chi_c * Bnu)), 1e-14)
check("pure continuum: no Q, U, V emission", np.abs(e[1:]).max(), 1e-300)
check("pure continuum: no polarized propagation terms", np.abs(h[1:]).max(), 1e-300)

# 5. Line plus continuum: additive in eta_I and j_I, polarized parts untouched.
J = jtensor(1.0, -0.1)
for B, th, ch, C in [(0.0, 0.0, 0.0, 0.0), (40.0, 1.0, 0.4, 1e8)]:
    e0, h0 = local(J, B, th, ch, C_ul=C)
    e1, h1 = local(J, B, th, ch, C_ul=C, chi_c=chi_c)
    check(f"eta_I gains chi_c (B={B:g})", np.abs(h1[0] - h0[0] - chi_c).max() / chi_c, 1e-9)
    check(f"j_I gains chi_c B_nu (B={B:g})", np.abs(e1[0] - e0[0] - chi_c * Bnu).max() / (chi_c * Bnu), 1e-9)
    check(f"polarized emission and propagation unchanged (B={B:g})",
          max(np.abs(e1[1:] - e0[1:]).max() / np.abs(e0[0]).max(), np.abs(h1[1:] - h0[1:]).max() / np.abs(h0[0]).max()), 1e-12)

# 5b. Total source function S = (eta_l S_l + chi_c B) / (eta_l + chi_c).
e1, h1 = local(J0, C_ul=1e8, chi_c=chi_c)
eps = destruction_probability(LINE, T, 1e8)
_, hl = local(J0, C_ul=1e8)
S_l = (1 - eps) + eps * Bnu
expected = (hl[0] * S_l + chi_c * Bnu) / (hl[0] + chi_c)
check("total source function S = (eta_l S_l + chi_c B)/(eta_l + chi_c)", rel(e1[0] / h1[0], expected), 1e-12)

# 6. Unresolved-field wrapper passes the continuum through unchanged.
args = (J, LINE, MU, PHI, NU, T, 1.0)
eU0, hU0 = point_emission_and_propagation_unresolved(
    *args, 20.0, 1.0, 0.4, 20.0, 0.0, VMIC, 0.0, 1e8, gamma=np.pi / 2, n_theta=6, n_chi=6)
eU1, hU1 = point_emission_and_propagation_unresolved(
    *args, 20.0, 1.0, 0.4, 20.0, 0.0, VMIC, 0.0, 1e8, gamma=np.pi / 2, n_theta=6, n_chi=6, chi_c=chi_c)
check("unresolved average: eta_I gains chi_c", np.abs(hU1[0] - hU0[0] - chi_c).max() / chi_c, 1e-9)
check("unresolved average: j_I gains chi_c B_nu", np.abs(eU1[0] - eU0[0] - chi_c * Bnu).max() / (chi_c * Bnu), 1e-9)

# 7. Atmosphere and fill_grid: chi_c default, validation, and per-depth use.
n = 3


def atmosphere(**kw):
    return Atmosphere1D(
        z=np.array([2.0, 1.0, 0.0]), T=np.array([4500.0, 6000.0, 7500.0]), B=np.zeros(n),
        theta_B=np.zeros(n), chi_B=np.zeros(n), v_los=np.zeros(n), D2=np.zeros(n),
        vmic=np.full(n, VMIC), **kw)


check("chi_c defaults to zero", np.abs(atmosphere().chi_c).max(), 1e-300)
for bad in ({"chi_c": np.array([0.0, -1e-6, 0.0])}, {"chi_c": np.zeros(2)}):
    try:
        atmosphere(**bad)
    except ValueError:
        print(f"[PASS] invalid chi_c rejected: {list(bad['chi_c'])}")
    else:
        raise AssertionError(f"invalid chi_c accepted: {bad}")

chi_depth = np.array([0.0, 2e-6, 8e-6])
line3 = LineModel(NU0, A_UL, 1.0, 1.0, 1, 0, 40, n_l=np.array([1.0, 2.0, 3.0]), C_ul=np.array([0.0, 1e7, 1e9]))
grid_c = Grid.build(atmosphere(chi_c=chi_depth), line3, n_mu_half=2, n_phi=4, n_nu=41).allocate()
grid_0 = Grid.build(atmosphere(), line3, n_mu_half=2, n_phi=4, n_nu=41).allocate()
for g in (grid_c, grid_0):
    g.JKQ[:, 0, 2] = 1.0
    fill_grid(g)
for k in range(n):
    Bk = planck(NU0, grid_c.atm.T[k])
    scale_eta = max(chi_depth[k], 1e-6)  # depth 0 has chi_c = 0: compare against a nominal scale
    check(f"fill_grid depth {k}: eta_I gains chi_c[k]",
          np.abs(grid_c.eta[k, 0] - grid_0.eta[k, 0] - chi_depth[k]).max() / scale_eta, 1e-9)
    check(f"fill_grid depth {k}: j_I gains chi_c[k] B_nu",
          np.abs(grid_c.emis[k, 0] - grid_0.emis[k, 0] - chi_depth[k] * Bk).max() / (scale_eta * Bk), 1e-9)
print("All continuum and destruction-probability checks passed.")
