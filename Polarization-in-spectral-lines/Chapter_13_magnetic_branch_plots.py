import os
import sys
import numpy as np
import matplotlib.pyplot as plt

script_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(script_dir)

from functions_prt import wigner_D2
from Radiation_fun import T, idx, Jrad_to_array, radiation_tensor
from Hanle_fun import hanle_parameter_exact
from Profile_fun import (
    Phi_generalized,
    Phi_appendix,
    damping_parameter,
    A_ul,
    default_Delta_nu_D,
)
from Rotation_fun import _los_vec, _angles_from_vec, _basis_from_angles, _rotate_vert_to_mag, _rotate_qu


# -----------------------------------------------------------------------------
# Global controls
# -----------------------------------------------------------------------------
B_GAUSS = 5.69
GJU = 1.0
HP = 0.073
XGRID = np.linspace(-5.0, 5.0, 401)

USE_Q_U_REFERENCE_MODE = "fixed_gamma_rotate_qu_back"
# Allowed values:
#   "transport_gamma"
#   "fixed_gamma_rotate_qu_back"

OUT_DIR = os.path.join(os.path.dirname(__file__), "Chapter_13_plots")

HU_VALUES_DASHED = [0.01, 0.08, 0.16, 0.25, 0.36, 0.50, 0.69, 0.98, 1.54, 3.16]
CHI_CONST_DEG_SOLID = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]
HU_GRID_SOLID = np.logspace(-6, np.log10(3.16), 400)
CHI_GRID_DASHED = np.linspace(0.0, 2.0 * np.pi, 721)

# For Hanle diagrams, the observer-frame equivalent magnetic operator
# H = D Hdiag D^dagger is more stable and maps directly to the book's
# pQ/pU definitions for fixed observer geometry.
HANLE_USE_VERTICAL_EQUIVALENT_OPERATOR = True

# Hanle diagram geometry mode:
#   "legacy" keeps the current direct T(theta_obs, chi_obs, gamma_obs) usage.
#   "ll04_eq1318" uses the composite rotation from LL04 Eq. (13.18)
#   in the geometric tensor contraction.
#   "ll04_full_chain" applies LL04 Eq. (13.14) + (13.17) + (13.18)
#   together using the magnetic-frame rho^2_Q convention.
#   "ll04_strict_single_frame" uses a strict single-frame LL04 contraction:
#   J(hR) in vertical frame -> rotate to magnetic frame -> Hanle response ->
#   Eq. (13.18)-style geometric tensors for I,Q,U with full epsI denominator.
HANLE_DIAGRAM_GEOMETRY_MODE = "legacy"

# Optional: use profile-integrated polarization (~pQ, ~pU), analogous to
# dashed lines in Fig. 13.7, for selected Hanle-diagram figures.
HANLE_USE_INTEGRATED_FORM = False
HANLE_INTEGRATED_PROFILE_KIND = "generalized"  # "generalized" or "appendix"
HANLE_INTEGRATED_TARGETS = {"Fig13_4", "Fig13_5"}


# -----------------------------------------------------------------------------
# Small helpers
# -----------------------------------------------------------------------------
def fmt_num(val, ndigits=6):
    return f"{val:.{ndigits}g}".replace("+", "p").replace("-", "m")


def deg_tag(rad):
    return fmt_num(np.degrees(rad), ndigits=4)


def ensure_out_dir(path):
    os.makedirs(path, exist_ok=True)


def _trapz(y, x):
    if hasattr(np, "trapezoid"):
        return np.trapezoid(y, x)
    return np.trapz(y, x)

def geometric_tensor_q_ll04(i, delta, theta_B, chi_B, gamma_B, theta_obs, chi_obs, gamma_obs):
    """
    LL04 Eq. (13.17)-(13.18):
      T^2_Q(i, Omega_0) = sum_P t^2_P(i) D^2_{P Q}(R),
      R = (-90deg, -90deg + delta, 0) x (chi_B, theta_B, gamma_B).
    """
    tP = {}
    for p in [-2, -1, 0, 1, 2]:
        tP[p] = T(i, 2, p, theta_obs, chi_obs, gamma_obs)

    dtot = (
        wigner_D2(-np.pi / 2, -np.pi / 2 + delta, 0.0)
        @ wigner_D2(chi_B, theta_B, gamma_B)
    )

    tq = {}
    for q in [-2, -1, 0, 1, 2]:
        s = 0.0 + 0.0j
        for p in [-2, -1, 0, 1, 2]:
            s += tP[p] * dtot[idx(p), idx(q)]
        tq[q] = s

    return tq


def radiation_tensor_delta(hp, delta):
    """
    Same projected-height construction used in the Chapter-13 notebook script.
    """
    hR = (1.0 + hp) / np.cos(delta) - 1.0
    j0 = radiation_tensor(hR)

    jarr = np.zeros(5, dtype=complex)
    jarr[idx(0)] = j0[(2, 0)]

    dmat = wigner_D2(0.0, delta, 0.0)
    jrot = dmat @ jarr

    jout = {(0, 0): j0[(0, 0)]}
    for q in [-2, -1, 0, 1, 2]:
        jout[(2, q)] = jrot[idx(q)]

    return j0, hR


# -----------------------------------------------------------------------------
# Magnetic-branch state preparation (no old branch here)
# -----------------------------------------------------------------------------
def _observer_angles_with_reference(los, qref, pole_tol=1e-10):
    los = np.asarray(los, dtype=float)
    los /= np.linalg.norm(los)

    qref = np.asarray(qref, dtype=float)
    qref -= np.dot(qref, los) * los
    qref_norm = np.linalg.norm(qref)
    if qref_norm <= pole_tol:
        raise ValueError("The transported Q reference is parallel to the LOS.")
    qref /= qref_norm

    if np.hypot(los[0], los[1]) <= pole_tol:
        theta = 0.0 if los[2] >= 0.0 else np.pi
        chi = 0.0
    else:
        theta, chi = _angles_from_vec(los)

    e_theta, e_chi = _basis_from_angles(theta, chi)
    gamma = np.arctan2(np.dot(qref, e_chi), np.dot(qref, e_theta))
    return theta, chi, gamma


def prepare_magnetic_branch_state(
    jrad,
    hu,
    theta_B,
    chi_B,
    theta_obs,
    chi_obs,
    gamma_obs,
    q_u_reference_mode,
    pole_tol=1e-10,
):
    jarr_base = Jrad_to_array(jrad)
    j00_base = jrad[(0, 0)]

    theta_obs_vert = theta_obs
    chi_obs_vert = chi_obs
    gamma_obs_vert = gamma_obs

    e_th_v, e_ch_v = _basis_from_angles(theta_obs_vert, chi_obs_vert)
    qref_vert = np.cos(gamma_obs_vert) * e_th_v + np.sin(gamma_obs_vert) * e_ch_v

    los_vert = _los_vec(theta_obs_vert, chi_obs_vert)
    los_mag = _rotate_vert_to_mag(los_vert, theta_B, chi_B)
    qref_mag = _rotate_vert_to_mag(qref_vert, theta_B, chi_B)

    qref_mag = qref_mag - np.dot(qref_mag, los_mag) * los_mag
    qref_norm = np.linalg.norm(qref_mag)
    if qref_norm <= pole_tol:
        raise ValueError("The transported Q reference is parallel to the LOS.")
    qref_mag = qref_mag / qref_norm

    transverse_norm = np.hypot(los_mag[0], los_mag[1])
    if transverse_norm <= pole_tol:
        # The LOS azimuth is undefined at a pole; use a fixed local chart.
        theta_mag = 0.0 if los_mag[2] >= 0.0 else np.pi
        chi_mag = 0.0
    else:
        theta_mag, chi_mag = _angles_from_vec(los_mag)

    theta_mag, chi_mag, gamma_transport = _observer_angles_with_reference(
        los_mag,
        qref_mag,
        pole_tol=pole_tol,
    )

    if q_u_reference_mode == "transport_gamma":
        gamma_mag = gamma_transport
        qu_back_rotation = 0.0
    elif q_u_reference_mode == "fixed_gamma_rotate_qu_back":
        gamma_mag = gamma_obs_vert
        qu_back_rotation = gamma_transport - gamma_obs_vert
    else:
        raise ValueError(f"Unknown Q_U_REFERENCE_MODE: {q_u_reference_mode}")

    dmag = wigner_D2(chi_B, theta_B, 0.0)
    jmag = dmag.conj().T @ jarr_base

    rho2_base = np.zeros(5, dtype=complex)
    for q in [-2, -1, 0, 1, 2]:
        rho2_base[idx(q)] = jmag[idx(q)] / (1.0 + 1j * q * hu)

    return {
        "J00": j00_base,
        "rho2": rho2_base,
        "theta_obs": theta_mag,
        "chi_obs": chi_mag,
        "gamma_obs": gamma_mag,
        "qu_back_rotation": qu_back_rotation,
    }


# -----------------------------------------------------------------------------
# Hanle-diagram values using magnetic branch
# -----------------------------------------------------------------------------
def hanle_point_pq_pu(
    hu,
    jrad,
    theta_B,
    chi_B,
    theta_obs,
    chi_obs,
    gamma_obs,
    q_u_reference_mode,
    delta=0.0,
    integrated_phi=None,
):
    if integrated_phi is not None and HANLE_DIAGRAM_GEOMETRY_MODE not in ["ll04_full_chain", "ll04_eq1318", "ll04_strict_single_frame"]:
        state = prepare_magnetic_branch_state(
            jrad,
            hu,
            theta_B,
            chi_B,
            theta_obs,
            chi_obs,
            gamma_obs,
            q_u_reference_mode,
        )
        return integrated_pq_pu_from_state(state, integrated_phi)

    if HANLE_DIAGRAM_GEOMETRY_MODE == "ll04_full_chain":
        # Full LL04 chain: Eq. (13.14) for J->magnetic frame, Eq. (10.27)-style
        # rho^2_Q magnetic-frame response, and Eq. (13.17)-(13.18) for geometry.
        jarr = Jrad_to_array(jrad)
        j00 = jrad[(0, 0)]

        d_b = wigner_D2(chi_B, theta_B, 0.0)
        jmag = d_b @ jarr

        rho_mag = np.zeros(5, dtype=complex)
        for q in [-2, -1, 0, 1, 2]:
            rho_mag[idx(q)] = ((-1) ** q) * jmag[idx(-q)] / (1.0 + 1j * q * hu)

        tq_geom = geometric_tensor_q_ll04(
            1,
            delta,
            theta_B,
            chi_B,
            0.0,
            theta_obs,
            chi_obs,
            gamma_obs,
        )
        tu_geom = geometric_tensor_q_ll04(
            2,
            delta,
            theta_B,
            chi_B,
            0.0,
            theta_obs,

            chi_obs,
            gamma_obs,
        )

        epsQ = 0.0 + 0.0j
        epsU = 0.0 + 0.0j
        if integrated_phi is not None:
            ti_geom = geometric_tensor_q_ll04(
                0,
                delta,
                theta_B,
                chi_B,
                0.0,
                theta_obs,
                chi_obs,
                gamma_obs,
            )

            epsI = integrated_phi[(0, 0, 0)] * j00
            for q in [-2, -1, 0, 1, 2]:
                phase = (-1) ** q
                phi22 = integrated_phi[(2, 2, q)]
                rhoq = rho_mag[idx(-q)]
                epsI += phase * phi22 * ti_geom[q] * rhoq
                epsQ += phase * phi22 * tq_geom[q] * rhoq
                epsU += phase * phi22 * tu_geom[q] * rhoq

            return np.real(epsQ / epsI), np.real(epsU / epsI)

        for q in [-2, -1, 0, 1, 2]:
            epsQ += ((-1) ** q) * tq_geom[q] * rho_mag[idx(-q)]
            epsU += ((-1) ** q) * tu_geom[q] * rho_mag[idx(-q)]

        # For this normalized Hanle diagram we keep the same denominator as in
        # the LL04 diagnostic implementation used elsewhere in this repository.
        return np.real(epsQ / j00), np.real(epsU / j00)

    if HANLE_DIAGRAM_GEOMETRY_MODE == "ll04_eq1318":
        # Use J at true height in the vertical frame, then apply LL04 Eq. (13.18)
        # through the geometric tensor rather than absorbing delta in J components.
        jarr = Jrad_to_array(jrad)
        j00 = jrad[(0, 0)]

        d_b = wigner_D2(chi_B, theta_B, 0.0)
        jmag = d_b @ jarr

        rho_mag = np.zeros(5, dtype=complex)
        for q in [-2, -1, 0, 1, 2]:
            rho_mag[idx(q)] = jmag[idx(q)] / (1.0 + 1j * q * hu)

        tq_geom = geometric_tensor_q_ll04(
            1,
            delta,
            theta_B,
            chi_B,
            0.0,
            theta_obs,
            chi_obs,
            gamma_obs,
        )
        tu_geom = geometric_tensor_q_ll04(
            2,
            delta,
            theta_B,
            chi_B,
            0.0,
            theta_obs,
            chi_obs,
            gamma_obs,
        )

        epsQ = 0.0 + 0.0j
        epsU = 0.0 + 0.0j
        if integrated_phi is not None:
            ti_geom = geometric_tensor_q_ll04(
                0,
                delta,
                theta_B,
                chi_B,
                0.0,
                theta_obs,
                chi_obs,
                gamma_obs,
            )

            epsI = integrated_phi[(0, 0, 0)] * j00
            for q in [-2, -1, 0, 1, 2]:
                phase = (-1) ** q
                phi22 = integrated_phi[(2, 2, q)]
                rhoq = rho_mag[idx(-q)]
                epsI += phase * phi22 * ti_geom[q] * rhoq
                epsQ += phase * phi22 * tq_geom[q] * rhoq
                epsU += phase * phi22 * tu_geom[q] * rhoq

            return np.real(epsQ / epsI), np.real(epsU / epsI)

        for q in [-2, -1, 0, 1, 2]:
            epsQ += ((-1) ** q) * tq_geom[q] * rho_mag[idx(-q)]
            epsU += ((-1) ** q) * tu_geom[q] * rho_mag[idx(-q)]

        return np.real(epsQ / j00), np.real(epsU / j00)

    if HANLE_DIAGRAM_GEOMETRY_MODE == "ll04_strict_single_frame":
        # Strict single-frame LL04 path:
        #   1) Build J at true height hR(delta) in vertical frame (handled in main).
        #   2) Rotate J to magnetic frame (Eq. 13.14).
        #   3) Apply Hanle response in magnetic frame.
        #   4) Contract with Eq. (13.18)-style geometric tensors for I,Q,U,
        #      using the full epsI denominator.
        jarr = Jrad_to_array(jrad)
        j00 = jrad[(0, 0)]

        d_b = wigner_D2(chi_B, theta_B, 0.0)
        jmag = d_b @ jarr

        rho_mag = np.zeros(5, dtype=complex)
        for q in [-2, -1, 0, 1, 2]:
            rho_mag[idx(q)] = jmag[idx(q)] / (1.0 + 1j * q * hu)

        ti_geom = geometric_tensor_q_ll04(
            0,
            delta,
            theta_B,
            chi_B,
            0.0,
            theta_obs,
            chi_obs,
            gamma_obs,
        )
        tq_geom = geometric_tensor_q_ll04(
            1,
            delta,
            theta_B,
            chi_B,
            0.0,
            theta_obs,
            chi_obs,
            gamma_obs,
        )
        tu_geom = geometric_tensor_q_ll04(
            2,
            delta,
            theta_B,
            chi_B,
            0.0,
            theta_obs,
            chi_obs,
            gamma_obs,
        )

        epsI = j00 + 0.0j
        epsQ = 0.0 + 0.0j
        epsU = 0.0 + 0.0j

        if integrated_phi is not None:
            epsI = integrated_phi[(0, 0, 0)] * j00
            for q in [-2, -1, 0, 1, 2]:
                phase = (-1) ** q
                phi22 = integrated_phi[(2, 2, q)]
                rhoq = rho_mag[idx(-q)]
                epsI += phase * phi22 * ti_geom[q] * rhoq
                epsQ += phase * phi22 * tq_geom[q] * rhoq
                epsU += phase * phi22 * tu_geom[q] * rhoq

            return np.real(epsQ / epsI), np.real(epsU / epsI)

        for q in [-2, -1, 0, 1, 2]:
            phase = (-1) ** q
            rhoq = rho_mag[idx(-q)]
            epsI += phase * ti_geom[q] * rhoq
            epsQ += phase * tq_geom[q] * rhoq
            epsU += phase * tu_geom[q] * rhoq

        return np.real(epsQ / epsI), np.real(epsU / epsI)

    if HANLE_USE_VERTICAL_EQUIVALENT_OPERATOR:
        jarr = Jrad_to_array(jrad)
        j00 = jrad[(0, 0)]

        qvals = np.array([-2, -1, 0, 1, 2])
        hdiag = np.diag([1.0 / (1.0 + 1j * q * hu) for q in qvals])
        dmat = wigner_D2(chi_B, theta_B, 0.0)
        rho2 = (dmat @ hdiag @ dmat.conj().T) @ jarr

        epsI = j00 + 0j
        epsQ = 0.0 + 0j
        epsU = 0.0 + 0j

        for q in [-2, -1, 0, 1, 2]:
            phase = (-1) ** q
            rhoq = np.conj(rho2[idx(-q)])

            epsI += phase * T(0, 2, q, theta_obs, chi_obs, gamma_obs) * rhoq
            epsQ += phase * T(1, 2, q, theta_obs, chi_obs, gamma_obs) * rhoq
            epsU += phase * T(2, 2, q, theta_obs, chi_obs, gamma_obs) * rhoq

        return np.real(epsQ / epsI), np.real(epsU / epsI)

    state = prepare_magnetic_branch_state(
        jrad,
        hu,
        theta_B,
        chi_B,
        theta_obs,
        chi_obs,
        gamma_obs,
        q_u_reference_mode,
    )
    for q in [-2, -1, 0, 1, 2]:
        phase = (-1) ** q
        rhoq = np.conj(state["rho2"][idx(-q)])

        epsI += phase * T(0, 2, q, state["theta_obs"], state["chi_obs"], state["gamma_obs"]) * rhoq
        epsQ += phase * T(1, 2, q, state["theta_obs"], state["chi_obs"], state["gamma_obs"]) * rhoq
        epsU += phase * T(2, 2, q, state["theta_obs"], state["chi_obs"], state["gamma_obs"]) * rhoq

    pQ = np.real(epsQ / epsI)
    pU = np.real(epsU / epsI)

    if np.abs(state["qu_back_rotation"]) > 0.0:
        pQ, pU = _rotate_qu(pQ, pU, state["qu_back_rotation"])

    return pQ, pU


def plot_hanle_diagram(
    fig_label,
    jrad,
    vH,
    theta_B,
    theta_obs,
    chi_obs,
    gamma_obs,
    out_dir,
    delta=0.0,
    integrated_phi=None,
):
    fig, ax = plt.subplots(figsize=(9, 9))

    # Dashed: Hu = const, chi_B varies
    for hu in HU_VALUES_DASHED:
        pu_curve = []
        pq_curve = []
        for chi_B in CHI_GRID_DASHED:
            pQ, pU = hanle_point_pq_pu(
                hu,
                jrad,
                theta_B,
                chi_B,
                theta_obs,
                chi_obs,
                gamma_obs,
                USE_Q_U_REFERENCE_MODE,
                delta,
                integrated_phi,
            )
            pu_curve.append(pU)
            pq_curve.append(pQ)

        ax.plot(pu_curve, pq_curve, "--", lw=1.2, alpha=0.85, label=f"Hu={hu:g}")

    # Solid: chi_B = const, Hu varies
    curve_number = {
        0: 1,
        30: 2,
        60: 3,
        90: 4,
        120: 5,
        150: 6,
        180: 7,
        210: 8,
        240: 9,
        270: 10,
        300: 11,
        330: 12,
    }

    for chi_deg in CHI_CONST_DEG_SOLID:
        pu_curve = []
        pq_curve = []
        chi_B = np.radians(chi_deg)

        for hu in HU_GRID_SOLID:
            pQ, pU = hanle_point_pq_pu(
                hu,
                jrad,
                theta_B,
                chi_B,
                theta_obs,
                chi_obs,
                gamma_obs,
                USE_Q_U_REFERENCE_MODE,
                delta,
                integrated_phi,
            )
            pu_curve.append(pU)
            pq_curve.append(pQ)

        ax.plot(pu_curve, pq_curve, "k-", lw=1.0)

        idx_lab = np.argmin(np.abs(HU_GRID_SOLID - curve_number[chi_deg]))
        ax.text(
            pu_curve[idx_lab],
            pq_curve[idx_lab],
            f"{curve_number[chi_deg]}",
            fontsize=8,
            ha="center",
            va="center",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75},
        )

    ax.set_xlabel("~pU")
    ax.set_ylabel("~pQ")
    ax.set_title(f"{fig_label} Hanle diagram (magnetic-frame branch)")
    ax.grid(alpha=0.3)
    ax.set_aspect("equal")
    ax.legend(loc="upper right", fontsize=7, ncol=1)

    hu_tag = f"HuGrid{fmt_num(np.min(HU_GRID_SOLID), 3)}to{fmt_num(np.max(HU_VALUES_DASHED), 3)}"
    chi_tag = "chiB0to360"
    fname = (
        f"{fig_label}_Hanle_vH{fmt_num(vH,6)}_{hu_tag}_"
        f"gamma{deg_tag(gamma_obs)}_thetaB{deg_tag(theta_B)}_{chi_tag}_delta{fmt_num(np.degrees(delta), 3)}.png"
    )
    out_path = os.path.join(out_dir, fname)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved: {out_path}")


# -----------------------------------------------------------------------------
# Stokes profiles (generalized / appendix) using magnetic branch
# -----------------------------------------------------------------------------
def build_phi_table(xgrid, profile_kind, vH, a_voigt):
    phi = {}
    for k in [0, 1, 2]:
        for kp in [0, 1, 2]:
            for q in [-2, -1, 0, 1, 2]:
                if profile_kind == "generalized":
                    phi[(k, kp, q)] = Phi_generalized(xgrid, k, kp, q, vH, a_voigt)
                elif profile_kind == "appendix":
                    phi[(k, kp, q)] = Phi_appendix(xgrid, k, kp, q, vH, a_voigt)
                else:
                    raise ValueError(f"Unknown profile kind: {profile_kind}")
    return phi


def build_integrated_phi_table(xgrid, profile_kind, vH, a_voigt):
    phi = build_phi_table(xgrid, profile_kind, vH, a_voigt)
    phi_int = {}
    for key, values in phi.items():
        phi_int[key] = _trapz(values, xgrid)
    return phi_int


def integrated_pq_pu_from_state(state, phi_int):
    epsI = 0.0 + 0.0j
    epsQ = 0.0 + 0.0j
    epsU = 0.0 + 0.0j

    j00 = state["J00"]
    rho2 = state["rho2"]
    theta = state["theta_obs"]
    chi = state["chi_obs"]
    gamma = state["gamma_obs"]

    # K=0 blocks
    epsI += phi_int[(0, 0, 0)] * T(0, 0, 0, theta, chi, gamma) * j00

    phi02 = phi_int[(0, 2, 0)]
    epsI += phi02 * T(0, 2, 0, theta, chi, gamma) * j00
    epsQ += phi02 * T(1, 2, 0, theta, chi, gamma) * j00
    epsU += phi02 * T(2, 2, 0, theta, chi, gamma) * j00

    # K=2 blocks
    for q in [-2, -1, 0, 1, 2]:
        phase = (-1) ** q
        rhoq = np.conj(rho2[idx(-q)])

        epsI += phase * phi_int[(2, 0, q)] * T(0, 0, 0, theta, chi, gamma) * rhoq

        phi21 = phi_int[(2, 1, q)]
        epsI += phase * phi21 * T(0, 1, q, theta, chi, gamma) * rhoq
        epsQ += phase * phi21 * T(1, 1, q, theta, chi, gamma) * rhoq
        epsU += phase * phi21 * T(2, 1, q, theta, chi, gamma) * rhoq

        phi22 = phi_int[(2, 2, q)]
        epsI += phase * phi22 * T(0, 2, q, theta, chi, gamma) * rhoq
        epsQ += phase * phi22 * T(1, 2, q, theta, chi, gamma) * rhoq
        epsU += phase * phi22 * T(2, 2, q, theta, chi, gamma) * rhoq

    pQ = np.real(epsQ / epsI)
    pU = np.real(epsU / epsI)

    if np.abs(state["qu_back_rotation"]) > 0.0:
        pQ, pU = _rotate_qu(pQ, pU, state["qu_back_rotation"])

    return pQ, pU


def compute_stokes_profiles(xgrid, phi, state):
    ip = np.zeros_like(xgrid)
    qp = np.zeros_like(xgrid)
    up = np.zeros_like(xgrid)
    vp = np.zeros_like(xgrid)

    j00 = state["J00"]
    #print(f"J00 = {j00}")
    rho2 = state["rho2"]
    #print(f"rho2 = {rho2}")
    theta = state["theta_obs"]
    chi = state["chi_obs"]
    gamma = state["gamma_obs"]

    for ix, _x in enumerate(xgrid):
        epsI = 0.0 + 0j
        epsQ = 0.0 + 0j
        epsU = 0.0 + 0j
        epsV = 0.0 + 0j

        # K=0 blocks
        phi00 = phi[(0, 0, 0)][ix]
        epsI += phi00 * T(0, 0, 0, theta, chi, gamma) * j00

        phi01 = phi[(0, 1, 0)][ix]
        epsV += phi01 * T(3, 1, 0, theta, chi, gamma) * j00

        phi02 = phi[(0, 2, 0)][ix]
        epsI += phi02 * T(0, 2, 0, theta, chi, gamma) * j00
        epsQ += phi02 * T(1, 2, 0, theta, chi, gamma) * j00
        epsU += phi02 * T(2, 2, 0, theta, chi, gamma) * j00

        # K=2 blocks
        for q in [-2, -1, 0, 1, 2]:
            phase = (-1) ** q
            rhoq = np.conj(rho2[idx(-q)])

            phi20 = phi[(2, 0, q)][ix]
            epsI += phase * phi20 * T(0, 0, 0, theta, chi, gamma) * rhoq

            phi21 = phi[(2, 1, q)][ix]
            epsI += phase * phi21 * T(0, 1, q, theta, chi, gamma) * rhoq
            epsQ += phase * phi21 * T(1, 1, q, theta, chi, gamma) * rhoq
            epsU += phase * phi21 * T(2, 1, q, theta, chi, gamma) * rhoq
            epsV += phase * phi21 * T(3, 1, q, theta, chi, gamma) * rhoq

            phi22 = phi[(2, 2, q)][ix]
            epsI += phase * phi22 * T(0, 2, q, theta, chi, gamma) * rhoq
            epsQ += phase * phi22 * T(1, 2, q, theta, chi, gamma) * rhoq
            epsU += phase * phi22 * T(2, 2, q, theta, chi, gamma) * rhoq

        i_val = np.real(epsI) * np.sqrt(np.pi)
        q_val = np.real(epsQ) * np.sqrt(np.pi)
        u_val = np.real(epsU) * np.sqrt(np.pi)
        v_val = np.real(epsV) * np.sqrt(np.pi)

        if np.abs(state["qu_back_rotation"]) > 0.0:
            q_val, u_val = _rotate_qu(q_val, u_val, state["qu_back_rotation"])

        ip[ix] = i_val
        qp[ix] = q_val
        up[ix] = u_val
        vp[ix] = v_val

    return ip, qp, up, vp


def plot_stokes(xgrid, I, Q, U, V, title, save_path):
    fig, ax = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(title)

    ax[0, 0].plot(xgrid, I)
    ax[0, 0].set_title("I")

    ax[0, 1].plot(xgrid, Q)
    ax[0, 1].set_title("Q")

    ax[1, 0].plot(xgrid, U)
    ax[1, 0].set_title("U")

    ax[1, 1].plot(xgrid, V)
    ax[1, 1].set_title("V")

    for aa in ax.ravel():
        aa.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)


def plot_fractional_polarization(x, I, Q, U, V, title, save_path):
    safe_I = np.where(np.abs(I) > 1e-300, I, np.nan)
    pQ = Q / safe_I
    pU = U / safe_I
    pV = V / safe_I

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, pQ, color="tab:blue", linewidth=2.0, label="pQ = Q/I")
    ax.plot(x, pU, color="tab:orange", linewidth=2.0, label="pU = U/I")
    ax.plot(x, pV, color="tab:green", linewidth=2.0, label="pV = V/I")

    intI = _trapz(I, x)
    if np.abs(intI) > 1e-300:
        pQ_tilde = _trapz(Q, x) / intI
        pU_tilde = _trapz(U, x) / intI
        pV_tilde = _trapz(V, x) / intI

        ax.axhline(pQ_tilde, color="tab:blue", linestyle="--", alpha=0.8, label="~pQ")
        ax.axhline(pU_tilde, color="tab:orange", linestyle="--", alpha=0.8, label="~pU")
        ax.axhline(pV_tilde, color="tab:green", linestyle="--", alpha=0.8, label="~pV")

    ax.set_xlabel("Reduced frequency x")
    ax.set_ylabel("Fractional polarization")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(save_path, dpi=250)
    plt.close(fig)


def run_stokes_figure_set(
    fig_label,
    jrad,
    hu,
    vH,
    a_voigt,
    theta_B,
    chi_B,
    theta_obs,
    chi_obs,
    gamma_obs,
    out_dir,
):
    state = prepare_magnetic_branch_state(
        jrad,
        hu,
        theta_B,
        chi_B,
        theta_obs,
        chi_obs,
        gamma_obs,
        USE_Q_U_REFERENCE_MODE,
    )

    for profile_kind in ["generalized", "appendix"]:
        phi = build_phi_table(XGRID, profile_kind, vH, a_voigt)
        I, Q, U, V = compute_stokes_profiles(XGRID, phi, state)

        base_tag = (
            f"{fig_label}_{profile_kind}_vH{fmt_num(vH,6)}_Hu{fmt_num(hu,6)}_"
            f"gamma{deg_tag(gamma_obs)}_thetaB{deg_tag(theta_B)}_chiB{deg_tag(chi_B)}"
        )

        stokes_path = os.path.join(out_dir, f"{base_tag}_Stokes.png")
        frac_path = os.path.join(out_dir, f"{base_tag}_Fractional.png")

        plot_stokes(
            XGRID,
            I,
            Q,
            U,
            V,
            (
                f"{fig_label} ({profile_kind}) | "
                f"Hu={hu:.6g}, vH={vH:.6g}, "
                f"theta_B={np.degrees(theta_B):.2f} deg, "
                f"chi_B={np.degrees(chi_B):.2f} deg"
            ),
            stokes_path,
        )

        plot_fractional_polarization(
            XGRID,
            I,
            Q,
            U,
            V,
            f"{fig_label} fractional ({profile_kind})",
            frac_path,
        )

        print(f"Saved: {stokes_path}")
        print(f"Saved: {frac_path}")


# -----------------------------------------------------------------------------
# Main run
# -----------------------------------------------------------------------------
def main():
    ensure_out_dir(OUT_DIR)

    a_voigt = damping_parameter()
    hu_default = hanle_parameter_exact(B_GAUSS, GJU, A_ul)
    vH_default = 1.3996e6 * B_GAUSS / default_Delta_nu_D

    print("Magnetic-branch-only Chapter 13 plotting")
    print("OUT_DIR =", OUT_DIR)
    print("a_voigt =", a_voigt)
    print("Hu_default =", hu_default)
    print("vH_default =", vH_default)
    print("Q/U reference mode =", USE_Q_U_REFERENCE_MODE)
    print("Hanle geometry mode =", HANLE_DIAGRAM_GEOMETRY_MODE)
    print("Hanle integrated-form mode =", HANLE_USE_INTEGRATED_FORM)
    if HANLE_USE_INTEGRATED_FORM:
        print("Hanle integrated profile kind =", HANLE_INTEGRATED_PROFILE_KIND)

    # -----------------------------------------------------------------
    # Hanle diagrams: Fig. 13.3, Fig. 13.4, Fig. 13.5
    # -----------------------------------------------------------------
    hanle_fig_configs = {
        # Fig. 13.3: delta = 0 deg, theta_B = 90 deg
        "Fig13_3": {
            "delta": 0.0,
            "theta_B": np.pi / 2,
            "theta_obs": np.pi / 2,
            "chi_obs": 0.0,
            "gamma_obs": np.pi / 2,
        },
        # Fig. 13.4: delta = +30 deg, theta_B = 90 deg
        "Fig13_4": {
            "delta": np.radians(30.0),
            "theta_B": np.pi / 2,
            "theta_obs": np.radians(90.0 - 30.0),  # theta_obs = 90 - delta
            "chi_obs": 0.0,
            "gamma_obs": np.pi / 2,
        },
        # Fig. 13.5 (book caption): delta = +30 deg, theta_B = 45 deg
        "Fig13_5": {
            "delta": np.radians(30.0),
            "theta_B": np.pi / 4,
            "theta_obs": np.radians(90.0 - 30.0),  # theta_obs = 90 - delta
            "chi_obs": 0.0,
            "gamma_obs": np.pi / 2,
        },
    }

    hanle_integrated_phi = None
    if HANLE_USE_INTEGRATED_FORM:
        hanle_integrated_phi = build_integrated_phi_table(
            XGRID,
            HANLE_INTEGRATED_PROFILE_KIND,
            vH_default,
            a_voigt,
        )

    for fig_label, cfg in hanle_fig_configs.items():
        delta = cfg["delta"]

        # Eq. (13.18) path uses delta in the geometric tensor; keep J in the
        # vertical frame at true height hR(delta).
        if HANLE_DIAGRAM_GEOMETRY_MODE in ["ll04_eq1318", "ll04_full_chain", "ll04_strict_single_frame"]:
            hR_delta = (1.0 + HP) / np.cos(delta) - 1.0
            jrad_delta = radiation_tensor(hR=hR_delta)
        else:
            jrad_delta, hR_delta = radiation_tensor_delta(HP, delta)

        print(
            f"{fig_label}: delta={np.degrees(delta):.1f} deg, "
            f"theta_B={np.degrees(cfg['theta_B']):.1f} deg, hR={hR_delta:.6f}"
        )

        plot_hanle_diagram(
            fig_label=fig_label,
            jrad=jrad_delta,
            vH=vH_default,
            theta_B=cfg["theta_B"],
            theta_obs=cfg["theta_obs"],
            chi_obs=cfg["chi_obs"],
            gamma_obs=cfg["gamma_obs"],
            out_dir=OUT_DIR,
            delta=delta,
            integrated_phi=(
                hanle_integrated_phi
                if fig_label in HANLE_INTEGRATED_TARGETS
                else None
            ),
        )

    # -----------------------------------------------------------------
    # Stokes / fractional: Fig. 13.6, Fig. 13.7, Fig. 13.8
    # -----------------------------------------------------------------
    jrad_base = radiation_tensor(hR=HP)

    # You can edit these geometry presets if you want to exactly match a
    # specific published panel definition.
    fig_geometries = {
        # Common Fig. 13.6-like setup (horizontal field in the POS).
        "Fig13_6": {
            "theta_B": np.pi / 2,
            "chi_B": 0.0,
            "theta_obs": np.pi / 2,
            "chi_obs": 0.0,
            "gamma_obs": np.pi / 2,
        },
        # Fig. 13.7 is fractional polarization representation of same setup.
        "Fig13_7": {
            "theta_B": np.pi / 2,
            "chi_B": 0.0,
            "theta_obs": np.pi / 2,
            "chi_obs": 0.0,
            "gamma_obs": np.pi / 2,
        },
        # Fig. 13.8-like oblique setup used in your recent Stokes script work.
        "Fig13_8": {
            "theta_B": np.pi / 4,
            "chi_B": -np.pi / 2,
            "theta_obs": np.pi / 2,
            "chi_obs": 0.0,
            "gamma_obs": np.pi / 2,
        },
    }

    for fig_label, cfg in fig_geometries.items():
        run_stokes_figure_set(
            fig_label=fig_label,
            jrad=jrad_base,
            hu=hu_default,
            vH=vH_default,
            a_voigt=a_voigt,
            theta_B=cfg["theta_B"],
            chi_B=cfg["chi_B"],
            theta_obs=cfg["theta_obs"],
            chi_obs=cfg["chi_obs"],
            gamma_obs=cfg["gamma_obs"],
            out_dir=OUT_DIR,
        )

    print("Done.")


if __name__ == "__main__":
    main()
