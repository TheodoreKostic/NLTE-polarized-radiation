import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

script_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(script_dir)

from Hanle_fun import hanle_parameter_exact
from Radiation_fun import radiation_tensor
from Profile_fun import damping_parameter, A_ul, default_Delta_nu_D
from Chapter_13_magnetic_branch_plots import (
    prepare_magnetic_branch_state,
    build_phi_table,
    compute_stokes_profiles,
    hanle_point_pq_pu,
    HU_VALUES_DASHED,
    CHI_CONST_DEG_SOLID,
    HU_GRID_SOLID,
    CHI_GRID_DASHED,
    USE_Q_U_REFERENCE_MODE,
    ensure_out_dir,
    fmt_num,
    HP,
)

# -----------------------------------------------------------------------------
# User controls
# -----------------------------------------------------------------------------
OUT_DIR = os.path.join(os.path.dirname(__file__), "Overview_plots")
XGRID = np.linspace(-5.0, 5.0, 401)
PROFILE_KIND = "generalized"  # "generalized" or "appendix"

B_GAUSS = 5.69
GJU = 1.0

# (J_0^0, J_0^2) pairs; the other J_Q^2 components are zero (axial symmetry).
J_VALUES = [
    (1.0, 0.0),
    (1.0, -0.1),
    (1.0, 0.1),
]

# (label, theta_B [deg], chi_B [deg]) magnetic field orientations.
FIELD_ORIENTATIONS = [
    ("B_parallel_z", 0.0, 0.0),
    ("B_horizontal_chi0", 90.0, 0.0),
    ("B_horizontal_chi90", 90.0, 90.0),
    ("B_oblique_45_chim90", 45.0, -90.0),
    ("B_oblique_45_chi45", 45.0, 45.0),
]

LABEL_TEX = {
    "B_parallel_z": r"$\mathbf{B}\parallel z$",
    "B_horizontal_chi0": r"$\mathbf{B}$ horizontal, $\chi_B=0^\circ$",
    "B_horizontal_chi90": r"$\mathbf{B}$ horizontal, $\chi_B=90^\circ$",
    "B_oblique_45_chim90": r"$\mathbf{B}$ oblique, $\theta_B=45^\circ$, $\chi_B=-90^\circ$",
    "B_oblique_45_chi45": r"$\mathbf{B}$ oblique, $\theta_B=45^\circ$, $\chi_B=45^\circ$",
}

# Combined overplot figure: (B [G], chi_B [deg]) cases, one figure per theta_B.
COMBINED_CASES = [(0.91, 0.0), (5.69, 0.0), (5.69, 90.0), (5.69, 180.0), (18.0, 45.0)]
COMBINED_THETA_B_DEG = [90.0, 60.0, 45.0, 30.0]
COMBINED_STRENGTHS_B_GAUSS = [0.0, 1.0, 2.0, 4.0, 10.0, 20.0, 50.0]
COMBINED_STRENGTHS_CHI_B_DEG = [0.0, 90.0, 180.0, -45.0, 0.0, 90.0, 180.0]
HU_GRID_STRENGTH = np.geomspace(
    1e-6,
    hanle_parameter_exact(COMBINED_STRENGTHS_B_GAUSS[-1], GJU, A_ul),
    400,
)
ABSORPTION_SCALE = 3.0

THETA_OBS_DEG_CASES = [90.0, 60.0]
CHI_OBS = 0.0
GAMMA_OBS = np.pi / 2


def b_from_hu(hu):
    return hu * A_ul / (2.0 * np.pi * 1.3996e6 * GJU)


def make_jrad(j00, j20):
    jrad = {(0, 0): complex(j00)}
    for q in [-2, -1, 0, 1, 2]:
        jrad[(2, q)] = complex(j20) if q == 0 else 0.0 + 0.0j
    return jrad


def draw_hanle_grid(ax, jrad, theta_B, theta_obs):
    def point(h, chi):
        return hanle_point_pq_pu(
            h, jrad, theta_B, chi, theta_obs, CHI_OBS, GAMMA_OBS,
            USE_Q_U_REFERENCE_MODE,
        )

    for h in HU_VALUES_DASHED:
        pq, pu = zip(*[point(h, c) for c in CHI_GRID_DASHED])
        ax.plot(pu, pq, "--", lw=1.0, alpha=0.8, label=rf"$B={b_from_hu(h):.3g}$ G")

    for chi_deg in CHI_CONST_DEG_SOLID:
        pq, pu = zip(*[point(h, np.radians(chi_deg)) for h in HU_GRID_SOLID])
        ax.plot(pu, pq, "k-", lw=0.8)
        ax.annotate(
            rf"${chi_deg}^\circ$", (pu[-1], pq[-1]), fontsize=7,
            xytext=(3, 3), textcoords="offset points",
        )

    ax.set_xlabel(r"$\tilde{p}_U$")
    ax.set_ylabel(r"$\tilde{p}_Q$")
    ax.set_title("Hanle diagram")
    ax.grid(alpha=0.3)
    ax.set_aspect("equal", adjustable="datalim")
    return point


def plot_hanle(ax, jrad, theta_B, chi_B, theta_obs, hu):
    point = draw_hanle_grid(ax, jrad, theta_B, theta_obs)
    pq0, pu0 = point(hu, chi_B)
    ax.plot(
        pu0, pq0, "ro", ms=7, zorder=5,
        label=rf"$B={B_GAUSS:g}$ G, $\chi_B={np.degrees(chi_B):.0f}^\circ$",
    )
    ax.legend(loc="upper right", fontsize=7)


def make_combined_figure(
    j00, j20, theta_B, theta_obs, a_voigt, cases=COMBINED_CASES, name_tag=""
):
    jrad = make_jrad(j00, j20)
    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(3, 2, width_ratios=[1, 1.5])
    axes_s = [fig.add_subplot(gs[i, 0]) for i in range(3)]
    ax_h = fig.add_subplot(gs[:, 1])
    point = draw_hanle_grid(ax_h, jrad, theta_B, theta_obs)

    for k, (b_gauss, chi_deg) in enumerate(cases):
        color = f"C{k}"
        chi_B = np.radians(chi_deg)
        hu_k = hanle_parameter_exact(b_gauss, GJU, A_ul)
        vH_k = 1.3996e6 * b_gauss / default_Delta_nu_D
        phi_k = build_phi_table(XGRID, PROFILE_KIND, vH_k, a_voigt)
        state = prepare_magnetic_branch_state(
            jrad=jrad, hu=hu_k, theta_B=theta_B, chi_B=chi_B,
            theta_obs=theta_obs, chi_obs=CHI_OBS, gamma_obs=GAMMA_OBS,
            q_u_reference_mode=USE_Q_U_REFERENCE_MODE,
        )
        I, Q, U, _ = compute_stokes_profiles(XGRID, phi_k, state)
        lab = rf"$B={b_gauss:g}$ G, $\chi_B={chi_deg:g}^\circ$"
        for ax, y in zip(axes_s, (1.0 - ABSORPTION_SCALE * I, Q, U)):
            ax.plot(XGRID, y, color=color, label=lab)
        pq, pu = point(hu_k, chi_B)
        ax_h.plot(pu, pq, "o", color=color, ms=8, zorder=5, mec="k", label=lab)

    for ax, name in zip(axes_s, (r"$I$", r"$Q$", r"$U$")):
        ax.set_ylabel(name)
        ax.grid(alpha=0.3)
        ax.legend(loc="best", fontsize=6)
    axes_s[-1].set_xlabel(r"Reduced frequency $x$")
    ax_h.legend(loc="upper right", fontsize=7)

    fig.suptitle(
        rf"$J^0_0={j00:g}$, $J^2_0={j20:g}$ | "
        rf"$\theta_B={np.degrees(theta_B):.1f}^\circ$, "
        rf"$\theta_{{obs}}={np.degrees(theta_obs):.1f}^\circ$"
    )
    fig.tight_layout()
    path = os.path.join(
        OUT_DIR,
        f"Overview_combined{name_tag}_thetaObs{fmt_num(np.degrees(theta_obs), 4)}_"
        f"thetaB{fmt_num(np.degrees(theta_B), 4)}.png",
    )
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print("Saved:", path)


def make_strength_sweep_figure(j00, j20, theta_B, theta_obs, a_voigt):
    jrad = make_jrad(j00, j20)
    colors = plt.get_cmap("tab10")(np.linspace(0.0, 1.0, len(COMBINED_STRENGTHS_B_GAUSS)))
    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(3, 2, width_ratios=[1, 1.5])
    axes_s = [fig.add_subplot(gs[i, 0]) for i in range(3)]
    ax_h = fig.add_subplot(gs[:, 1])

    def point(hu_value, chi_value):
        return hanle_point_pq_pu(
            hu_value, jrad, theta_B, chi_value, theta_obs, CHI_OBS,
            GAMMA_OBS, USE_Q_U_REFERENCE_MODE,
        )

    chi_guide_endpoints = []
    for chi_index, chi_deg in enumerate(CHI_CONST_DEG_SOLID):
        pq, pu = zip(*[
            point(hu_value, np.radians(chi_deg))
            for hu_value in HU_GRID_STRENGTH
        ])
        ax_h.plot(pu, pq, "k-", lw=0.7, alpha=0.75, zorder=1)
        chi_guide_endpoints.append((chi_deg, pu[-1], pq[-1]))

    b_legend_handles = []
    dot_legend_handles = []
    b_legend_handles = []
    for b_index, (b_gauss, color) in enumerate(zip(COMBINED_STRENGTHS_B_GAUSS, colors)):
        hu_value = hanle_parameter_exact(b_gauss, GJU, A_ul)
        vH = 1.3996e6 * b_gauss / default_Delta_nu_D
        phi = build_phi_table(XGRID, PROFILE_KIND, vH, a_voigt)
        chi_deg = COMBINED_STRENGTHS_CHI_B_DEG[b_index]
        chi_B = np.radians(chi_deg)

        chi_curve = [point(hu_value, chi_value) for chi_value in CHI_GRID_DASHED]
        pq_curve, pu_curve = zip(*chi_curve)
        if b_gauss == 0.0:
            ax_h.plot(pu_curve[0], pq_curve[0], "--", color=color, lw=1.5,
                      label=rf"$B={b_gauss:g}$ G", zorder=3)
        else:
            ax_h.plot(pu_curve, pq_curve, "--", color=color, lw=1.5,
                      label=rf"$B={b_gauss:g}$ G", zorder=3)

        b_legend_handles.append(
            Line2D([0], [0], color=color, lw=2, linestyle="--", label=rf"$B={b_gauss:g}$ G")
        )

        state = prepare_magnetic_branch_state(
            jrad=jrad, hu=hu_value, theta_B=theta_B, chi_B=chi_B,
            theta_obs=theta_obs, chi_obs=CHI_OBS, gamma_obs=GAMMA_OBS,
            q_u_reference_mode=USE_Q_U_REFERENCE_MODE,
        )
        I, Q, U, _ = compute_stokes_profiles(XGRID, phi, state)
        lab = rf"$B={b_gauss:g}$ G, $\chi_B={chi_deg:.1f}^\circ$"
        for ax, profile in zip(axes_s, (1.0 - ABSORPTION_SCALE * I, Q, U)):
            ax.plot(XGRID, profile, color=color, lw=1.2, label=lab)

        pQ, pU = point(hu_value, chi_B)
        ax_h.plot(pU, pQ, "o", color=color, mec="k", mew=0.5, ms=6, zorder=5)
        dot_legend_handles.append(
            Line2D(
                [0], [0], marker="o", color="none", markerfacecolor=color,
                markeredgecolor="k", markersize=5,
                label=rf"$B={b_gauss:g}$ G, $\chi_B={chi_deg:g}^\circ$",
            )
        )

    for chi_deg, pu_endpoint, pq_endpoint in chi_guide_endpoints:
        ax_h.annotate(
            rf"${chi_deg}^\circ$", (pu_endpoint, pq_endpoint),
            fontsize=8, fontweight="bold",
            xytext=(4, 4), textcoords="offset points",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 0.5},
        )

    for ax, name in zip(axes_s, (r"$I$", r"$Q$", r"$U$")):
        ax.set_ylabel(name)
        ax.grid(alpha=0.3)
        ax.legend(loc="upper right" if name == r"$U$" else "best", fontsize=6)
    axes_s[-1].set_xlabel(r"Reduced frequency $x$")
    ax_h.set_xlabel(r"$\tilde{p}_U$")
    ax_h.set_ylabel(r"$\tilde{p}_Q$")
    ax_h.set_title("Hanle diagram")
    ax_h.grid(alpha=0.3)
    ax_h.set_aspect("equal", adjustable="datalim")
    b_legend = ax_h.legend(
        handles=b_legend_handles, loc="upper right", fontsize=6, framealpha=0.9
    )
    ax_h.add_artist(b_legend)
    ax_h.legend(
        handles=dot_legend_handles, loc="upper right", bbox_to_anchor=(1.0, 0.68),
        fontsize=6, framealpha=0.9,
    )

    fig.suptitle(
        rf"$J^0_0={j00:g}$, $J^2_0={j20:g}$ | "
        rf"$\theta_B={np.degrees(theta_B):.1f}^\circ$, "
        rf"$\theta_{{obs}}={np.degrees(theta_obs):.1f}^\circ$"
    )
    fig.tight_layout(rect=(0, 0.15, 1, 0.96))
    path = os.path.join(
        OUT_DIR,
        f"Overview_combined_Bstrengths_thetaObs{fmt_num(np.degrees(theta_obs), 4)}_"
        f"thetaB{fmt_num(np.degrees(theta_B), 4)}.png",
    )
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print("Saved:", path)


def make_figure(jrad, j00, j20, label, theta_B, chi_B, theta_obs, phi, hu):
    state = prepare_magnetic_branch_state(
        jrad=jrad,
        hu=hu,
        theta_B=theta_B,
        chi_B=chi_B,
        theta_obs=theta_obs,
        chi_obs=CHI_OBS,
        gamma_obs=GAMMA_OBS,
        q_u_reference_mode=USE_Q_U_REFERENCE_MODE,
    )
    I, Q, U, _ = compute_stokes_profiles(XGRID, phi, state)

    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(3, 2, width_ratios=[1, 1.5])
    axes_s = [fig.add_subplot(gs[i, 0]) for i in range(3)]
    for ax, y, name in zip(axes_s, (I, Q, U), ("I", "Q", "U")):
        ax.plot(XGRID, y)
        ax.set_ylabel(rf"${name}$")
        ax.grid(alpha=0.3)
    axes_s[-1].set_xlabel(r"Reduced frequency $x$")

    ax_h = fig.add_subplot(gs[:, 1])
    plot_hanle(ax_h, jrad, theta_B, chi_B, theta_obs, hu)

    fig.suptitle(
        rf"$J^0_0={j00:g}$, $J^2_0={j20:g}$ | {LABEL_TEX.get(label, label)} | "
        rf"$B={B_GAUSS:g}$ G, $\theta_B={np.degrees(theta_B):.1f}^\circ$, "
        rf"$\chi_B={np.degrees(chi_B):.1f}^\circ$, "
        rf"$\theta_{{obs}}={np.degrees(theta_obs):.1f}^\circ$"
    )
    fig.tight_layout()

    fname = (
        f"Overview_thetaObs{fmt_num(np.degrees(theta_obs), 4)}_"
        f"J00_{fmt_num(j00, 4)}_J20_{fmt_num(j20, 4)}_{label}_"
        f"thetaB{fmt_num(np.degrees(theta_B), 4)}_chiB{fmt_num(np.degrees(chi_B), 4)}.png"
    )
    path = os.path.join(OUT_DIR, fname)
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print("Saved:", path)


def main():
    ensure_out_dir(OUT_DIR)
    a_voigt = damping_parameter()
    hu = hanle_parameter_exact(B_GAUSS, GJU, A_ul)
    vH = 1.3996e6 * B_GAUSS / default_Delta_nu_D
    phi = build_phi_table(XGRID, PROFILE_KIND, vH, a_voigt)

    j_hp = radiation_tensor(HP)
    j_cases = J_VALUES + [(float(np.real(j_hp[(0, 0)])), float(np.real(j_hp[(2, 0)])))]

    for theta_obs_deg in THETA_OBS_DEG_CASES:
        for th_b_deg in COMBINED_THETA_B_DEG:
            make_combined_figure(
                *j_cases[-1], np.radians(th_b_deg),
                np.radians(theta_obs_deg), a_voigt,
            )
            make_strength_sweep_figure(
                *j_cases[-1], np.radians(th_b_deg),
                np.radians(theta_obs_deg), a_voigt,
            )
        for j00, j20 in j_cases:
            jrad = make_jrad(j00, j20)
            for label, th_deg, chi_deg in FIELD_ORIENTATIONS:
                make_figure(
                    jrad, j00, j20, label,
                    np.radians(th_deg), np.radians(chi_deg),
                    np.radians(theta_obs_deg), phi, hu,
                )


if __name__ == "__main__":
    main()
