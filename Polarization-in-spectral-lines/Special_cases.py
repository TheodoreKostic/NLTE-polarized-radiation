import os
import sys
import numpy as np

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


def main():
    ensure_out_dir(OUT_DIR)

    jrad = radiation_tensor(hR=HP)
    a_voigt = damping_parameter()

    print("Running special-case Stokes profiles")
    print("Output directory:", OUT_DIR)
    print("Profile kind:", PROFILE_KIND)
    print("Observer chi, gamma [deg] =", np.degrees(CHI_OBS), np.degrees(GAMMA_OBS))

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

    print(f"Done. Generated {total} Stokes plots.")


if __name__ == "__main__":
    main()
