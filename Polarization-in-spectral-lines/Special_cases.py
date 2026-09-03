import os
import sys
import numpy as np
import matplotlib.pyplot as plt

script_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(script_dir)

from Hanle_fun import hanle_parameter_exact
from Profile_fun import damping_parameter, A_ul, default_Delta_nu_D
from Radiation_fun import radiation_tensor

from Chapter_13_magnetic_branch_plots import (
    prepare_magnetic_branch_state,
    build_phi_table,
    compute_stokes_profiles,
    plot_stokes,
    plot_hanle_diagram,
    hanle_point_pq_pu,
    build_integrated_phi_table,
    HU_VALUES_DASHED,
    CHI_CONST_DEG_SOLID,
    HU_GRID_SOLID,
    CHI_GRID_DASHED,
    USE_Q_U_REFERENCE_MODE,
    ensure_out_dir,
    fmt_num,
)


# -----------------------------------------------------------------------------
# User controls
# -----------------------------------------------------------------------------
OUT_DIR = os.path.join(os.path.dirname(__file__), "Special_cases_plots")
XGRID = np.linspace(-5.0, 5.0, 401)
PROFILE_KIND = "generalized"  # "generalized" or "appendix"

HP = 0.073
B_DEFAULT_GAUSS = 5.69
GJU = 1.0

CHI_OBS = 0.0
GAMMA_OBS = np.pi / 2


def observer_theta_cases():
    return {
        "thetaObs_90deg": np.pi / 2,
        "thetaObs_60deg": np.pi / 3,
        "thetaObs_0deg": 0.0,
    }


def make_case_list(theta_obs):
    return [
        {
            "case": "Case1_B0",
            "description": "No magnetic field",
            "B": 0.0,
            "theta_B": 0.0,
            "chi_B": 0.0,
        },
        {
            "case": "Case2_B_parallel_z",
            "description": "B parallel to z",
            "B": B_DEFAULT_GAUSS,
            "theta_B": 0.0,
            "chi_B": 0.0,
        },
        {
            "case": "Case3_B_perp_z",
            "description": "B orthogonal to z",
            "B": B_DEFAULT_GAUSS,
            "theta_B": np.pi / 2,
            "chi_B": 0.0,
        },
        {
            "case": "Case4_B_perp_z_perp_Omega",
            "description": "B orthogonal to z and to observer",
            "B": B_DEFAULT_GAUSS,
            "theta_B": np.pi / 2,
            "chi_B": np.pi / 2,
        },
    ]


def stokes_for_case(jrad, a_voigt, theta_obs, case_cfg, phi_cache):
    b_gauss = case_cfg["B"]
    hu = hanle_parameter_exact(b_gauss, GJU, A_ul)
    vH = 1.3996e6 * b_gauss / default_Delta_nu_D

    if vH not in phi_cache:
        phi_cache[vH] = build_phi_table(XGRID, PROFILE_KIND, vH, a_voigt)

    state = prepare_magnetic_branch_state(
        jrad=jrad,
        hu=hu,
        theta_B=case_cfg["theta_B"],
        chi_B=case_cfg["chi_B"],
        theta_obs=theta_obs,
        chi_obs=CHI_OBS,
        gamma_obs=GAMMA_OBS,
        q_u_reference_mode="fixed_gamma_rotate_qu_back",
    )

    return compute_stokes_profiles(XGRID, phi_cache[vH], state), hu, vH


def vH_from_hu(hu):
    return hu * A_ul / (2.0 * np.pi * GJU * default_Delta_nu_D)


def plot_hanle_curves(
    ax,
    jrad,
    theta_B,
    theta_obs,
    hu_fixed,
    chi_B_fixed,
    is_zero_field,
    integrated_phi_for_hu=None,
):
    def point(hu, chi_B):
        integrated_phi = (
            integrated_phi_for_hu(hu)
            if integrated_phi_for_hu is not None
            else None
        )
        return hanle_point_pq_pu(
            hu,
            jrad,
            theta_B,
            chi_B,
            theta_obs,
            CHI_OBS,
            GAMMA_OBS,
            USE_Q_U_REFERENCE_MODE,
            integrated_phi=integrated_phi,
        )

    if is_zero_field:
        pQ, pU = point(0.0, chi_B_fixed)
        ax.plot(pU, pQ, "ko", markersize=5, zorder=3)
        ax.grid(alpha=0.3)
        ax.set_aspect("equal", adjustable="box")
        return

    for hu in HU_VALUES_DASHED:
        curve = [
            point(hu, chi_B)
            for chi_B in CHI_GRID_DASHED
        ]
        pQ, pU = zip(*curve)
        ax.plot(pU, pQ, "--", lw=1.2, alpha=0.85)

    for chi_deg in CHI_CONST_DEG_SOLID:
        chi_B = np.radians(chi_deg)
        curve = [
            point(hu, chi_B)
            for hu in HU_GRID_SOLID
        ]
        pQ, pU = zip(*curve)
        ax.plot(pU, pQ, "k-", lw=1.0)

    pQ_fixed, pU_fixed = point(hu_fixed, chi_B_fixed)
    ax.plot(pU_fixed, pQ_fixed, "ko", markersize=5, zorder=3)

    ax.grid(alpha=0.3)
    ax.set_aspect("equal", adjustable="box")


def plot_special_cases_hanle_grid(jrad, out_dir, a_voigt=None):
    observer_cases = observer_theta_cases()
    case_configs = make_case_list(next(iter(observer_cases.values())))
    fig, axes = plt.subplots(
        len(observer_cases),
        len(case_configs),
        figsize=(16, 12),
        sharex=True,
        sharey=True,
    )
    integrated_phi_cache = {}

    def integrated_phi_for_hu(hu):
        if hu not in integrated_phi_cache:
            integrated_phi_cache[hu] = build_integrated_phi_table(
                XGRID,
                PROFILE_KIND,
                vH_from_hu(hu),
                a_voigt,
            )
        return integrated_phi_cache[hu]

    for row, (obs_label, theta_obs) in enumerate(observer_cases.items()):
        for column, case_cfg in enumerate(case_configs):
            ax = axes[row, column]
            hu_fixed = hanle_parameter_exact(case_cfg["B"], GJU, A_ul)
            plot_hanle_curves(
                ax,
                jrad,
                case_cfg["theta_B"],
                theta_obs,
                hu_fixed,
                case_cfg["chi_B"],
                np.isclose(case_cfg["B"], 0.0),
                integrated_phi_for_hu if a_voigt is not None else None,
            )

            if row == 0:
                ax.set_title(case_cfg["description"], fontsize=9)
            if column == 0:
                ax.set_ylabel(f"{obs_label}\n~pQ")
            if row == len(observer_cases) - 1:
                ax.set_xlabel("~pU")

    fig.suptitle("Special cases: Hanle diagrams", fontsize=14)
    fig.tight_layout()
    suffix = "_integrated" if a_voigt is not None else ""
    save_path = os.path.join(out_dir, f"Special_cases_Hanle_grid{suffix}.png")
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    print("Saved:", save_path)


def main():
    ensure_out_dir(OUT_DIR)

    jrad = radiation_tensor(hR=HP)
    a_voigt = damping_parameter()

    print("Running special-case Stokes profiles")
    print("Output directory:", OUT_DIR)
    print("Profile kind:", PROFILE_KIND)
    print("Observer chi, gamma [deg] =", np.degrees(CHI_OBS), np.degrees(GAMMA_OBS))

    plot_special_cases_hanle_grid(jrad, OUT_DIR)
    plot_special_cases_hanle_grid(jrad, OUT_DIR, a_voigt=a_voigt)

    phi_cache = {}
    total = 0

    for obs_label, theta_obs in observer_theta_cases().items():
        cases = make_case_list(theta_obs)

        if np.isclose(theta_obs, 0.0):
            print(
                "Note: for theta_obs = 0, horizontal-B cases are degenerate with respect to LOS perpendicularity."
            )

        for case_cfg in cases:
            (I, Q, U, V), hu, vH = stokes_for_case(
                jrad=jrad,
                a_voigt=a_voigt,
                theta_obs=theta_obs,
                case_cfg=case_cfg,
                phi_cache=phi_cache,
            )

            fname = (
                f"{obs_label}_{case_cfg['case']}_"
                f"B{fmt_num(case_cfg['B'], 6)}_"
                f"thetaB{fmt_num(np.degrees(case_cfg['theta_B']), 4)}_"
                f"chiB{fmt_num(np.degrees(case_cfg['chi_B']), 4)}_"
                f"Hu{fmt_num(hu, 6)}_vH{fmt_num(vH, 6)}.png"
            )
            save_path = os.path.join(OUT_DIR, fname)

            title = (
                f"{obs_label} | {case_cfg['description']} | "
                f"B={case_cfg['B']:.4g} G, "
                f"theta_obs={np.degrees(theta_obs):.1f} deg, "
                f"theta_B={np.degrees(case_cfg['theta_B']):.1f} deg, "
                f"chi_B={np.degrees(case_cfg['chi_B']):.1f} deg"
            )

            plot_stokes(XGRID, I, Q, U, V, title, save_path)
            total += 1
            print("Saved:", save_path)

            hanle_label = f"{obs_label}_{case_cfg['case']}"
            plot_hanle_diagram(
                fig_label=hanle_label,
                jrad=jrad,
                vH=vH,
                theta_B=case_cfg["theta_B"],
                theta_obs=theta_obs,
                chi_obs=CHI_OBS,
                gamma_obs=GAMMA_OBS,
                out_dir=OUT_DIR,
            )

    print(f"Done. Generated {total} Stokes plots.")


if __name__ == "__main__":
    main()
