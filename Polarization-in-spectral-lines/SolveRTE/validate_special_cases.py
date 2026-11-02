"""Reproduce the Special_cases.py Stokes profiles with the grid emissivity module.

Same inputs as Special_cases.main(): J^K_Q = radiation_tensor(hR=0.073), gamma = pi/2,
chi_obs = 0, theta_obs in {90, 60, 0} deg, four magnetic cases, "generalized" profile.
Run:  python validate_special_cases.py
Add --plot to write diagnostic figures.
"""
import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_HERE)
for _p in (_HERE, _PARENT, os.path.dirname(_PARENT)):
    if _p not in sys.path:
        sys.path.append(_p)

import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--plot", action="store_true", help="write diagnostic figures")
ARGS = parser.parse_args()
if ARGS.plot:
    import matplotlib.pyplot as plt

import Special_cases as S
from Profile_fun import A_ul, default_Delta_nu_D, damping_parameter
from Radiation_fun import radiation_tensor
from Hanle_fun import hanle_parameter_exact
from atmosphere import LineModel, C_LIGHT
from grid_opacity import point_emission_and_propagation, eta0, phi00_norm

NU0 = 4.3e14
LINE = LineModel(NU0, A_ul, S.GJU, S.GJU, 1, 0, 40)
VMIC = default_Delta_nu_D * C_LIGHT / NU0  # with T ~ 0 this gives Delta nu_D = 4e9 exactly
X = S.XGRID
NU = -X * default_Delta_nu_D  # nu - nu0, since Special_cases x = (nu0 - nu)/dnu_D
SCALE = phi00_norm() * default_Delta_nu_D / eta0(LINE, 1.0) * np.sqrt(np.pi)

jrad = radiation_tensor(hR=S.HP)
a_voigt = damping_parameter()
JKQ = np.zeros((3, 5), complex)
JKQ[0, 2] = jrad[(0, 0)]
for q in range(-2, 3):
    JKQ[2, q + 2] = jrad[(2, q)]

cache, worst, worst_v, n_checks = {}, 0.0, 0.0, 0
TOL_IQU = 1e-10
TOL_V = 1e-10


def check(name, error, tolerance):
    global n_checks
    n_checks += 1
    passed = np.isfinite(error) and error < tolerance
    print(f"[{'PASS' if passed else 'FAIL'}] {name}: {error:.2e} (tol {tolerance:.0e})")
    if not passed:
        raise AssertionError(
            f"{name}: error {error:.3e} is not below tolerance {tolerance:.3e}"
        )


for obs_label, th_obs in S.observer_theta_cases().items():
    cases = S.make_case_list(th_obs)
    if ARGS.plot:
        fig, axes = plt.subplots(len(cases), 4, figsize=(15, 10), sharex=True)
    for r, cfg in enumerate(cases):
        (I, Q, U, V), hu, vH = S.stokes_for_case(jrad, a_voigt, th_obs, cfg, cache)
        eps, _ = point_emission_and_propagation(
            JKQ, LINE, np.array([np.cos(th_obs)]), np.array([S.CHI_OBS]), NU, 1e-12, 1.0,
            cfg["B"], cfg["theta_B"], cfg["chi_B"], 0.0, VMIC, 0.0, 0.0, gamma=S.GAMMA_OBS)
        mine = eps[:, :, 0, 0] * SCALE
        ref = np.array([I, Q, U, V])
        err_iqu = np.abs(mine[:3] - ref[:3]).max() / np.abs(ref[0]).max()
        worst = max(worst, err_iqu)
        vmax = np.abs(ref[3]).max()
        intensity_scale = np.abs(ref[0]).max()
        if vmax > 1e-12 * intensity_scale:
            errv = np.abs(mine[3] - ref[3]).max() / vmax
            worst_v = max(worst_v, errv)
            errv_label = "relative"
        else:
            errv = np.abs(mine[3] - ref[3]).max() / intensity_scale
            errv_label = "absolute, scaled by max I"
        label = f"{obs_label} {cfg['case']}"
        check(f"{label}: I,Q,U", err_iqu, TOL_IQU)
        check(f"{label}: V ({errv_label})", errv, TOL_V)
        print(f"    Hu={hu:.3f}; max|V_ref|={vmax:.1e}; max|V_mine|={np.abs(mine[3]).max():.1e}")
        if ARGS.plot:
            for c, nm in enumerate("IQUV"):
                ax = axes[r, c]
                ax.plot(X, ref[c], "k", lw=2, label="Special_cases")
                ax.plot(X, mine[c], "r--", lw=1.2, label="grid_opacity")
                ax.set_title(f"{cfg['description']}: {nm}", fontsize=8)
                ax.grid(alpha=0.2)
            axes[r, 0].legend(fontsize=7)
    if ARGS.plot:
        fig.suptitle(f"{obs_label}: Special_cases vs grid emissivity (J from radiation_tensor, gamma=pi/2)")
        fig.tight_layout()
        fig.savefig(f"validate_special_cases_{obs_label}.png", dpi=100)
        plt.close(fig)

print(f"\n{n_checks}/{n_checks} checks passed; worst I,Q,U error={worst:.1e}; V error={worst_v:.1e}")
