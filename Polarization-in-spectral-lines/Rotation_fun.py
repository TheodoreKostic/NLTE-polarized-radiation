import numpy as np
import sys
import os
import matplotlib.pyplot as plt

script_dir = os.path.abspath("/home/Code/NLTE-polarized-radiation")
#script_dir = os.path.abspath("/home/teodor/Documents/Codes/NLTE-polarized-radiation")
sys.path.append(script_dir)

from functions_prt import wigner_D2, wigner_d2
from Radiation_fun import *
from Profile_fun import *
from Hanle_fun import *


def _los_vec(theta, chi):
    return np.array([
        np.sin(theta) * np.cos(chi),
        np.sin(theta) * np.sin(chi),
        np.cos(theta)
    ], dtype=float)


def _angles_from_vec(v):
    vv = v / np.linalg.norm(v)
    theta = np.arccos(np.clip(vv[2], -1.0, 1.0))
    chi = np.arctan2(vv[1], vv[0])
    return theta, chi


def _basis_from_angles(theta, chi):
    # Local spherical basis for propagation direction (theta, chi).
    e_theta = np.array([
        np.cos(theta) * np.cos(chi),
        np.cos(theta) * np.sin(chi),
        -np.sin(theta)
    ], dtype=float)
    e_chi = np.array([
        -np.sin(chi),
        np.cos(chi),
        0.0
    ], dtype=float)
    return e_theta, e_chi


def _rotate_vert_to_mag(v, theta_B, chi_B):
    # Inverse of Rz(chi_B) @ Ry(theta_B): Ry(-theta_B) @ Rz(-chi_B)
    cz = np.cos(-chi_B)
    sz = np.sin(-chi_B)
    rz = np.array([
        [cz, -sz, 0.0],
        [sz, cz, 0.0],
        [0.0, 0.0, 1.0]
    ])

    cy = np.cos(-theta_B)
    sy = np.sin(-theta_B)
    ry = np.array([
        [cy, 0.0, sy],
        [0.0, 1.0, 0.0],
        [-sy, 0.0, cy]
    ])

    return ry @ (rz @ v)


def _rotate_qu(q, u, psi):
    c2 = np.cos(2.0 * psi)
    s2 = np.sin(2.0 * psi)
    q_new = q * c2 + u * s2
    u_new = -q * s2 + u * c2
    return q_new, u_new


def plot_fractional_polarization(x, I, Q, U, V, title, save_path):
    safe_I = np.where(np.abs(I) > 1e-300, I, np.nan)
    pQ = Q / safe_I
    pU = U / safe_I
    pV = V / safe_I

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, pQ, color="tab:blue", linewidth=2.0, label="pQ = Q/I")
    ax.plot(x, pU, color="tab:orange", linewidth=2.0, label="pU = U/I")
    ax.plot(x, pV, color="tab:green", linewidth=2.0, label="pV = V/I")

    intI = np.trapz(I, x)
    if np.abs(intI) > 1e-300:
        pQ_tilde = np.trapz(Q, x) / intI
        pU_tilde = np.trapz(U, x) / intI
        pV_tilde = np.trapz(V, x) / intI

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

if __name__ == "__main__":
    # Example usage of the functions defined above.

    # False: current vertical-frame path (full Hanle operator)
    # True : LL04-style magnetic-frame contraction (single frame)
    USE_LL04_MAG_FRAME_BRANCH = True

    # Q/U reference handling when using magnetic-frame branch:
    # "transport_gamma"            -> transport +Q axis and use transported gamma directly
    # "fixed_gamma_rotate_qu_back" -> keep original gamma in contraction, rotate Q/U back after
    Q_U_REFERENCE_MODE = "fixed_gamma_rotate_qu_back"

    theta_B = np.pi/2 # Magnetic field inclination (radians)
    chi_B = 0.0   # Magnetic field azimuth (radians)
    theta_obs = np.pi/2 # Observer inclination (radians)
    chi_obs = 0.0   # Observer azimuth (radians)
    gamma_obs = np.pi/2 # +Q reference azimuth (radians)

    B = 5.69 # Magnetic field strength (Gauss)
    Jrad_0 = radiation_tensor(hR=0.073)
    Jarr_base = Jrad_to_array(Jrad_0)
    J00_base = Jrad_0[(0,0)]
    Hu = hanle_parameter_exact(5.69, 1.0, A_ul)
    vH = 1.3996e6 * B / default_Delta_nu_D
    qu_back_rotation = 0.0

    theta_obs_vert = theta_obs
    chi_obs_vert = chi_obs
    gamma_obs_vert = gamma_obs

    # Build the original +Q reference direction in the vertical frame,
    # then transport it into the magnetic frame to preserve Q/U signs.
    e_th_v, e_ch_v = _basis_from_angles(theta_obs_vert, chi_obs_vert)
    qref_vert = (
        np.cos(gamma_obs_vert) * e_th_v
        + np.sin(gamma_obs_vert) * e_ch_v
    )

    los_vert = _los_vec(theta_obs, chi_obs)
    los_mag = _rotate_vert_to_mag(los_vert, theta_B, chi_B)
    qref_mag = _rotate_vert_to_mag(qref_vert, theta_B, chi_B)

    theta_obs, chi_obs = _angles_from_vec(los_mag)

    # Recompute gamma in magnetic frame from transported +Q reference.
    e_th_m, e_ch_m = _basis_from_angles(theta_obs, chi_obs)
    qref_mag = qref_mag - np.dot(qref_mag, los_mag) * los_mag
    qref_mag = qref_mag / np.linalg.norm(qref_mag)
    gamma_transport = np.arctan2(
        np.dot(qref_mag, e_ch_m),
        np.dot(qref_mag, e_th_m)
    )

    if Q_U_REFERENCE_MODE == "transport_gamma":
        gamma_obs = gamma_transport
        qu_back_rotation = 0.0
    elif Q_U_REFERENCE_MODE == "fixed_gamma_rotate_qu_back":
        gamma_obs = gamma_obs_vert
        qu_back_rotation = gamma_transport - gamma_obs_vert
    else:
        raise ValueError("Unknown Q_U_REFERENCE_MODE: {}".format(Q_U_REFERENCE_MODE))

    Dmag = wigner_D2(chi_B, theta_B, 0.0)
    Jmag = Dmag.conj().T @ Jarr_base

    rho2_base = np.zeros(5, dtype=complex)
    for Q in [-2, -1, 0, 1, 2]:
        rho2_base[idx(Q)] = Jmag[idx(Q)] / (1.0 + 1j * Q * Hu)

    print("Frame branch: magnetic-frame Eq.13.20")
    print("theta_obs (mag frame) [deg] =", np.degrees(theta_obs))
    print("chi_obs   (mag frame) [deg] =", np.degrees(chi_obs))
    print("Q/U reference mode =", Q_U_REFERENCE_MODE)
    print("gamma_obs (mag frame) [deg] =", np.degrees(gamma_obs))