import numpy as np
import sys
import os
import matplotlib.pyplot as plt

script_dir = os.path.abspath("/home/Code/NLTE-polarized-radiation")
#script_dir = os.path.abspath("/home/teodor/Documents/Codes/NLTE-polarized-radiation")
#script_dir = os.path.abspath("/home/mistflow/Documents/Doktorat/NLTE-polarized-radiation")
sys.path.append(script_dir)

from functions_prt import wigner_D2, wigner_d2
from Radiation_fun import *
from Hanle_fun import *
from Profile_fun import *
from Chapter_13_magnetic_branch_plots import *

def _vH_from_B(B_value):
    return 1.3996e6 * B_value / default_Delta_nu_D


def _angles_from_field_direction(field_direction):
    field_direction = np.asarray(field_direction, dtype=float)
    norm = np.linalg.norm(field_direction)
    if norm <= 0.0:
        raise ValueError("field_direction must be nonzero.")

    direction = field_direction / norm
    theta = np.arccos(np.clip(direction[2], -1.0, 1.0))
    chi = np.arctan2(direction[1], direction[0])
    return theta, chi


def directional_response_at_field_pole(
    stokes_from_angles,
    B_magnitude,
    epsilon,
    pole="north",
):
    if B_magnitude <= 0.0:
        raise ValueError("B_magnitude must be > 0.")
    if epsilon <= 0.0:
        raise ValueError("epsilon must be > 0.")
    if pole not in ("north", "south"):
        raise ValueError("pole must be 'north' or 'south'.")

    pole_direction = np.array([0.0, 0.0, 1.0 if pole == "north" else -1.0])
    tangent_directions = (
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
    )

    directional_responses = []
    for tangent in tangent_directions:
        direction_plus = (
            np.cos(epsilon) * pole_direction
            + np.sin(epsilon) * tangent
        )
        direction_minus = (
            np.cos(epsilon) * pole_direction
            - np.sin(epsilon) * tangent
        )

        theta_plus, chi_plus = _angles_from_field_direction(direction_plus)
        theta_minus, chi_minus = _angles_from_field_direction(direction_minus)

        stokes_plus = stokes_from_angles(theta_plus, chi_plus)
        stokes_minus = stokes_from_angles(theta_minus, chi_minus)
        directional_responses.append(tuple(
            (plus - minus) / (2.0 * epsilon)
            for plus, minus in zip(stokes_plus, stokes_minus)
        ))

    return directional_responses[0], directional_responses[1]


a_voigt = damping_parameter()
a = a_voigt

def B_finite_difference_response_local(
    xgrid,
    jrad,
    B0,
    delta_B,
    theta_B,
    chi_B,
    theta_obs,
    chi_obs,
    gamma_obs,
    q_u_reference_mode="fixed_gamma_rotate_qu_back",
    gJu=1.0,
    Aul=A_ul,
    profile_kind="generalized",
    a_value=None,
    scheme="central",
    normalize=None,   # None, "I", or "self"
    vary_hu_with_B=True,
    vary_vH_with_B=True,
):
    if delta_B <= 0.0:
        raise ValueError("delta_B must be > 0.")

    if a_value is None:
        a_value = a_voigt

    hu_ref = hanle_parameter_exact(B0, gJu, Aul)
    vH_ref = _vH_from_B(B0)

    # Recompute selected B-dependent terms while allowing dependency-isolation runs.
    def stokes_at_B(B_value):
        hu_value = hanle_parameter_exact(B_value, gJu, Aul) if vary_hu_with_B else hu_ref
        vH_value = _vH_from_B(B_value) if vary_vH_with_B else vH_ref
        phi_value = build_phi_table(
            xgrid,
            profile_kind=profile_kind,
            vH=vH_value,
            a_voigt=a_value,
        )
        state_value = prepare_magnetic_branch_state(
            jrad,
            hu_value,
            theta_B,
            chi_B,
            theta_obs,
            chi_obs,
            gamma_obs,
            q_u_reference_mode,
        )
        return compute_stokes_profiles(xgrid, phi_value, state_value)

    I0, Q0, U0, V0 = stokes_at_B(B0)
    Ip, Qp, Up, Vp = stokes_at_B(B0 + delta_B)

    if scheme == "central":
        Im, Qm, Um, Vm = stokes_at_B(B0 - delta_B)

        dIdB = (Ip - Im) / (2.0 * delta_B)
        dQdB = (Qp - Qm) / (2.0 * delta_B)
        dUdB = (Up - Um) / (2.0 * delta_B)
        dVdB = (Vp - Vm) / (2.0 * delta_B)
    elif scheme == "forward":
        dIdB = (Ip - I0) / delta_B
        dQdB = (Qp - Q0) / delta_B
        dUdB = (Up - U0) / delta_B
        dVdB = (Vp - V0) / delta_B
    else:
        raise ValueError("scheme must be 'central' or 'forward'.")

    # Optional normalization
    eps = 1e-300
    if normalize is None:
        return dIdB, dQdB, dUdB, dVdB, I0, Q0, U0, V0

    if normalize == "I":
        denom = np.where(np.abs(I0) > eps, I0, np.nan)
        return dIdB / denom, dQdB / denom, dUdB / denom, dVdB / denom, I0, Q0, U0, V0

    if normalize == "self":
        denI = np.where(np.abs(I0) > eps, I0, np.nan)
        denQ = np.where(np.abs(Q0) > eps, Q0, np.nan)
        denU = np.where(np.abs(U0) > eps, U0, np.nan)
        denV = np.where(np.abs(V0) > eps, V0, np.nan)
        return dIdB / denI, dQdB / denQ, dUdB / denU, dVdB / denV, I0, Q0, U0, V0

    raise ValueError("normalize must be None, 'I', or 'self'.")

def theta_B_finite_difference_response_local(
    xgrid,
    jrad,
    B_value,
    theta_B0,
    delta_theta_B,
    chi_B,
    theta_obs,
    chi_obs,
    gamma_obs,
    q_u_reference_mode="fixed_gamma_rotate_qu_back",
    gJu=1.0,
    Aul=A_ul,
    profile_kind="generalized",
    a_value=None,
    scheme="central",
    normalize=None,   # None, "I", or "self"
):
    if delta_theta_B <= 0.0:
        raise ValueError("delta_theta_B must be > 0.")

    if a_value is None:
        a_value = a_voigt

    # B (hence hu, vH) is fixed, so the profile table is independent of theta_B.
    hu_value = hanle_parameter_exact(B_value, gJu, Aul)
    vH_value = _vH_from_B(B_value)
    phi_value = build_phi_table(xgrid, profile_kind=profile_kind, vH=vH_value, a_voigt=a_value)

    def stokes_at_theta_B(theta_B_value):
        state_value = prepare_magnetic_branch_state(
            jrad,
            hu_value,
            theta_B_value,
            chi_B,
            theta_obs,
            chi_obs,
            gamma_obs,
            q_u_reference_mode,
        )
        return compute_stokes_profiles(xgrid, phi_value, state_value)

    I0, Q0, U0, V0 = stokes_at_theta_B(theta_B0)
    Ip, Qp, Up, Vp = stokes_at_theta_B(theta_B0 + delta_theta_B)

    if scheme == "central":
        Im, Qm, Um, Vm = stokes_at_theta_B(theta_B0 - delta_theta_B)

        dIdtheta = (Ip - Im) / (2.0 * delta_theta_B)
        dQdtheta = (Qp - Qm) / (2.0 * delta_theta_B)
        dUdtheta = (Up - Um) / (2.0 * delta_theta_B)
        dVdtheta = (Vp - Vm) / (2.0 * delta_theta_B)
    elif scheme == "forward":
        dIdtheta = (Ip - I0) / delta_theta_B
        dQdtheta = (Qp - Q0) / delta_theta_B
        dUdtheta = (Up - U0) / delta_theta_B
        dVdtheta = (Vp - V0) / delta_theta_B
    else:
        raise ValueError("scheme must be 'central' or 'forward'.")

    eps = 1e-300
    if normalize is None:
        return dIdtheta, dQdtheta, dUdtheta, dVdtheta, I0, Q0, U0, V0

    if normalize == "I":
        denom = np.where(np.abs(I0) > eps, I0, np.nan)
        return dIdtheta / denom, dQdtheta / denom, dUdtheta / denom, dVdtheta / denom, I0, Q0, U0, V0

    if normalize == "self":
        denI = np.where(np.abs(I0) > eps, I0, np.nan)
        denQ = np.where(np.abs(Q0) > eps, Q0, np.nan)
        denU = np.where(np.abs(U0) > eps, U0, np.nan)
        denV = np.where(np.abs(V0) > eps, V0, np.nan)
        return dIdtheta / denI, dQdtheta / denQ, dUdtheta / denU, dVdtheta / denV, I0, Q0, U0, V0

    raise ValueError("normalize must be None, 'I', or 'self'.")


def chi_B_finite_difference_response_local(
    xgrid,
    jrad,
    B_value,
    theta_B,
    chi_B0,
    delta_chi_B,
    theta_obs,
    chi_obs,
    gamma_obs,
    q_u_reference_mode="fixed_gamma_rotate_qu_back",
    gJu=1.0,
    Aul=A_ul,
    profile_kind="generalized",
    a_value=None,
    scheme="central",
    normalize=None,
):
    if delta_chi_B <= 0.0:
        raise ValueError("delta_chi_B must be > 0.")

    if a_value is None:
        a_value = a_voigt

    hu_value = hanle_parameter_exact(B_value, gJu, Aul)
    vH_value = _vH_from_B(B_value)
    phi_value = build_phi_table(xgrid, profile_kind=profile_kind, vH=vH_value, a_voigt=a_value)

    def stokes_at_chi_B(chi_B_value):
        state_value = prepare_magnetic_branch_state(
            jrad,
            hu_value,
            theta_B,
            chi_B_value,
            theta_obs,
            chi_obs,
            gamma_obs,
            q_u_reference_mode,
        )
        return compute_stokes_profiles(xgrid, phi_value, state_value)

    I0, Q0, U0, V0 = stokes_at_chi_B(chi_B0)
    Ip, Qp, Up, Vp = stokes_at_chi_B(chi_B0 + delta_chi_B)

    if scheme == "central":
        Im, Qm, Um, Vm = stokes_at_chi_B(chi_B0 - delta_chi_B)

        dIdchi = (Ip - Im) / (2.0 * delta_chi_B)
        dQdchi = (Qp - Qm) / (2.0 * delta_chi_B)
        dUdchi = (Up - Um) / (2.0 * delta_chi_B)
        dVdchi = (Vp - Vm) / (2.0 * delta_chi_B)
    elif scheme == "forward":
        dIdchi = (Ip - I0) / delta_chi_B
        dQdchi = (Qp - Q0) / delta_chi_B
        dUdchi = (Up - U0) / delta_chi_B
        dVdchi = (Vp - V0) / delta_chi_B
    else:
        raise ValueError("scheme must be 'central' or 'forward'.")

    eps = 1e-300
    if normalize is None:
        return dIdchi, dQdchi, dUdchi, dVdchi, I0, Q0, U0, V0

    if normalize == "I":
        denom = np.where(np.abs(I0) > eps, I0, np.nan)
        return dIdchi / denom, dQdchi / denom, dUdchi / denom, dVdchi / denom, I0, Q0, U0, V0

    if normalize == "self":
        denI = np.where(np.abs(I0) > eps, I0, np.nan)
        denQ = np.where(np.abs(Q0) > eps, Q0, np.nan)
        denU = np.where(np.abs(U0) > eps, U0, np.nan)
        denV = np.where(np.abs(V0) > eps, V0, np.nan)
        return dIdchi / denI, dQdchi / denQ, dUdchi / denU, dVdchi / denV, I0, Q0, U0, V0

    raise ValueError("normalize must be None, 'I', or 'self'.")

# ---------------------------------------------------------
# 1D response profiles at a single fixed height (fixed jrad)
# ---------------------------------------------------------
hR_fixed_1D = 0.073         # pick the height (hp/h_true value) you want to fix
jrad_fixed = radiation_tensor(hR_fixed_1D)

B0_1D = 5.69
delta_B_1D = 0.2
delta_theta_B_1D = np.radians(5.0)
delta_chi_B_1D = np.radians(5.0)

xgrid = np.linspace(-5.0, 5.0, 200)
theta_B = np.pi/2
chi_B = 0.0
theta_obs = np.pi/2
chi_obs = 0.0
gamma_obs = np.pi/2

profile_kind = "generalized"
Q_U_REFERENCE_MODE = "fixed_gamma_rotate_qu_back"

dIdB_1d, dQdB_1d, dUdB_1d, dVdB_1d, I0, Q0, U0, V0 = B_finite_difference_response_local(
    xgrid=xgrid,
    jrad=jrad_fixed,
    B0=B0_1D,
    delta_B=delta_B_1D,
    theta_B=theta_B,
    chi_B=chi_B,
    theta_obs=theta_obs,
    chi_obs=chi_obs,
    gamma_obs=gamma_obs,
    q_u_reference_mode=Q_U_REFERENCE_MODE,
    profile_kind=profile_kind,
    scheme="central",
    normalize=None,   # or "I" / "self" if you want fractional response
)

fig, ax = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
for a, resp, label in zip(
    ax.ravel(),
    [dIdB_1d, dQdB_1d, dUdB_1d, dVdB_1d],
    ["dI/dB", "dQ/dB", "dU/dB", "dV/dB"],
):
    a.plot(xgrid, resp)
    a.set_xlabel("Reduced frequency x")
    a.set_ylabel(label)
    a.grid(alpha=0.3)

fig.suptitle(f"Response to B at fixed h={hR_fixed_1D}, B0={B0_1D} G, delta_B={delta_B_1D} G")
fig.savefig(f"RF_1D_h{hR_fixed_1D}_B0{B0_1D}_delta_B{delta_B_1D}.png", dpi=300)
plt.close(fig)

dIdth, dQdth, dUdth, dVdth, *_ = theta_B_finite_difference_response_local(
    xgrid=xgrid,
    jrad=jrad_fixed,          # same fixed height/jrad 
    B_value=B0_1D,            # fixed field strength
    theta_B0=theta_B,
    delta_theta_B=delta_theta_B_1D,
    chi_B=chi_B,
    theta_obs=theta_obs,
    chi_obs=chi_obs,
    gamma_obs=gamma_obs,
    q_u_reference_mode=Q_U_REFERENCE_MODE,
    profile_kind=profile_kind,
    scheme="central",
    normalize=None,   # or "I" / "self" if you want fractional
)
fig, ax = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
for a, resp, label in zip(
    ax.ravel(),
    [dIdth, dQdth, dUdth, dVdth],
    ["dI/dtheta_B", "dQ/dtheta_B", "dU/dtheta_B", "dV/dtheta_B"],
):
    a.plot(xgrid, resp)
    a.set_xlabel("Reduced frequency x")
    a.set_ylabel(label)
    a.grid(alpha=0.3)   
fig.suptitle(f"Response to theta_B at fixed h={hR_fixed_1D}, B0={B0_1D} G, delta_theta_B=2 deg")
fig.savefig(f"RF_theta_B_1D_h{hR_fixed_1D}_B0{B0_1D}_delta_theta_B{int(np.degrees(delta_theta_B_1D))}deg.png", dpi=300)
plt.close(fig)

dIdchi, dQdchi, dUdchi, dVdchi, *_ = chi_B_finite_difference_response_local(
    xgrid=xgrid,
    jrad=jrad_fixed,
    B_value=B0_1D,
    theta_B=theta_B,
    chi_B0=chi_B,
    delta_chi_B=delta_chi_B_1D,
    theta_obs=theta_obs,
    chi_obs=chi_obs,
    gamma_obs=gamma_obs,
    q_u_reference_mode=Q_U_REFERENCE_MODE,
    profile_kind=profile_kind,
    scheme="central",
    normalize=None,   # or "I" / "self" if you want fractional
)
fig, ax = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
for a, resp, label in zip(
    ax.ravel(),
    [dIdchi, dQdchi, dUdchi, dVdchi],
    ["dI/dchi_B", "dQ/dchi_B", "dU/dchi_B", "dV/dchi_B"],
):
    a.plot(xgrid, resp)
    a.set_xlabel("Reduced frequency x")
    a.set_ylabel(label)
    a.grid(alpha=0.3)
fig.suptitle(f"Response to chi_B at fixed h={hR_fixed_1D}, B0={B0_1D} G, delta_chi_B={int(np.degrees(delta_chi_B_1D))} deg")
fig.savefig(f"RF_chi_B_1D_h{hR_fixed_1D}_B0{B0_1D}_delta_chi_B{int(np.degrees(delta_chi_B_1D))}deg.png", dpi=300)
plt.close(fig)

# ----------------------------------
# Derivative response functions using np.gradient (for comparison)
# ----------------------------------

def response_vs_B_gradient(
    xgrid,
    jrad,
    B_array,
    theta_B,
    chi_B,
    theta_obs,
    chi_obs,
    gamma_obs,
    q_u_reference_mode="fixed_gamma_rotate_qu_back",
    gJu=1.0,
    Aul=A_ul,
    profile_kind="generalized",
    a_value=None,
):
    if a_value is None:
        a_value = a_voigt

    n_b = len(B_array)
    n_x = len(xgrid)
    I_arr = np.zeros((n_b, n_x))
    Q_arr = np.zeros((n_b, n_x))
    U_arr = np.zeros((n_b, n_x))
    V_arr = np.zeros((n_b, n_x))

    # jrad (height) and observer geometry stay fixed across the whole B sweep.
    for ib, B_value in enumerate(B_array):
        hu_value = hanle_parameter_exact(B_value, gJu, Aul)
        vH_value = _vH_from_B(B_value)
        phi_value = build_phi_table(xgrid, profile_kind=profile_kind, vH=vH_value, a_voigt=a_value)
        state_value = prepare_magnetic_branch_state(
            jrad, hu_value, theta_B, chi_B, theta_obs, chi_obs, gamma_obs, q_u_reference_mode,
        )
        I_arr[ib], Q_arr[ib], U_arr[ib], V_arr[ib] = compute_stokes_profiles(xgrid, phi_value, state_value)

    dIdB = np.gradient(I_arr, B_array, axis=0)
    dQdB = np.gradient(Q_arr, B_array, axis=0)
    dUdB = np.gradient(U_arr, B_array, axis=0)
    dVdB = np.gradient(V_arr, B_array, axis=0)

    return dIdB, dQdB, dUdB, dVdB, I_arr, Q_arr, U_arr, V_arr


def response_vs_theta_B_gradient(
    xgrid,
    jrad,
    B_value,
    theta_B_array,
    chi_B,
    theta_obs,
    chi_obs,
    gamma_obs,
    q_u_reference_mode="fixed_gamma_rotate_qu_back",
    gJu=1.0,
    Aul=A_ul,
    profile_kind="generalized",
    a_value=None,
):
    if a_value is None:
        a_value = a_voigt

    # B (hence hu, vH) is fixed, so the profile table only needs to be built once.
    hu_value = hanle_parameter_exact(B_value, gJu, Aul)
    vH_value = _vH_from_B(B_value)
    phi_value = build_phi_table(xgrid, profile_kind=profile_kind, vH=vH_value, a_voigt=a_value)

    n_t = len(theta_B_array)
    n_x = len(xgrid)
    I_arr = np.zeros((n_t, n_x))
    Q_arr = np.zeros((n_t, n_x))
    U_arr = np.zeros((n_t, n_x))
    V_arr = np.zeros((n_t, n_x))

    for it, theta_B_value in enumerate(theta_B_array):
        state_value = prepare_magnetic_branch_state(
            jrad, hu_value, theta_B_value, chi_B, theta_obs, chi_obs, gamma_obs, q_u_reference_mode,
        )
        I_arr[it], Q_arr[it], U_arr[it], V_arr[it] = compute_stokes_profiles(xgrid, phi_value, state_value)

    dIdth = np.gradient(I_arr, theta_B_array, axis=0)
    dQdth = np.gradient(Q_arr, theta_B_array, axis=0)
    dUdth = np.gradient(U_arr, theta_B_array, axis=0)
    dVdth = np.gradient(V_arr, theta_B_array, axis=0)

    return dIdth, dQdth, dUdth, dVdth, I_arr, Q_arr, U_arr, V_arr


def response_vs_chi_B_gradient(
    xgrid,
    jrad,
    B_value,
    theta_B,
    chi_B_array,
    theta_obs,
    chi_obs,
    gamma_obs,
    q_u_reference_mode="fixed_gamma_rotate_qu_back",
    gJu=1.0,
    Aul=A_ul,
    profile_kind="generalized",
    a_value=None,
):
    if a_value is None:
        a_value = a_voigt

    hu_value = hanle_parameter_exact(B_value, gJu, Aul)
    vH_value = _vH_from_B(B_value)
    phi_value = build_phi_table(xgrid, profile_kind=profile_kind, vH=vH_value, a_voigt=a_value)

    n_c = len(chi_B_array)
    n_x = len(xgrid)
    I_arr = np.zeros((n_c, n_x))
    Q_arr = np.zeros((n_c, n_x))
    U_arr = np.zeros((n_c, n_x))
    V_arr = np.zeros((n_c, n_x))

    for ic, chi_B_value in enumerate(chi_B_array):
        state_value = prepare_magnetic_branch_state(
            jrad, hu_value, theta_B, chi_B_value, theta_obs, chi_obs, gamma_obs, q_u_reference_mode,
        )
        I_arr[ic], Q_arr[ic], U_arr[ic], V_arr[ic] = compute_stokes_profiles(xgrid, phi_value, state_value)

    dIdchi = np.gradient(I_arr, chi_B_array, axis=0)
    dQdchi = np.gradient(Q_arr, chi_B_array, axis=0)
    dUdchi = np.gradient(U_arr, chi_B_array, axis=0)
    dVdchi = np.gradient(V_arr, chi_B_array, axis=0)

    return dIdchi, dQdchi, dUdchi, dVdchi, I_arr, Q_arr, U_arr, V_arr

N_STEP = 5   # points on each side of the center

B_array = B0_1D + delta_B_1D * np.arange(-N_STEP, N_STEP + 1)
dIdB_g, dQdB_g, dUdB_g, dVdB_g, *_ = response_vs_B_gradient(
    xgrid, jrad_fixed, B_array, theta_B, chi_B, theta_obs, chi_obs, gamma_obs,
    q_u_reference_mode=Q_U_REFERENCE_MODE, profile_kind=profile_kind,
)
dIdB_center, dQdB_center, dUdB_center, dVdB_center = (
    dIdB_g[N_STEP], dQdB_g[N_STEP], dUdB_g[N_STEP], dVdB_g[N_STEP]
)

theta_B_array = theta_B + delta_theta_B_1D * np.arange(-N_STEP, N_STEP + 1)
dIdth_g, dQdth_g, dUdth_g, dVdth_g, *_ = response_vs_theta_B_gradient(
    xgrid, jrad_fixed, B0_1D, theta_B_array, chi_B, theta_obs, chi_obs, gamma_obs,
    q_u_reference_mode=Q_U_REFERENCE_MODE, profile_kind=profile_kind,
)
dIdth_center, dQdth_center, dUdth_center, dVdth_center = (
    dIdth_g[N_STEP], dQdth_g[N_STEP], dUdth_g[N_STEP], dVdth_g[N_STEP]
)

chi_B_array = chi_B + delta_chi_B_1D * np.arange(-N_STEP, N_STEP + 1)
dIdchi_g, dQdchi_g, dUdchi_g, dVdchi_g, *_ = response_vs_chi_B_gradient(
    xgrid, jrad_fixed, B0_1D, theta_B, chi_B_array, theta_obs, chi_obs, gamma_obs,
    q_u_reference_mode=Q_U_REFERENCE_MODE, profile_kind=profile_kind,
)
dIdchi_center, dQdchi_center, dUdchi_center, dVdchi_center = (
    dIdchi_g[N_STEP], dQdchi_g[N_STEP], dUdchi_g[N_STEP], dVdchi_g[N_STEP]
)

fig, ax = plt.subplots(3, 4, figsize=(18, 12), constrained_layout=True)
for a, resp, label in zip(
    ax.ravel(),
    [dIdB_center, dQdB_center, dUdB_center, dVdB_center,
     dIdth_center, dQdth_center, dUdth_center, dVdth_center,
     dIdchi_center, dQdchi_center, dUdchi_center, dVdchi_center],
    ["dI/dB", "dQ/dB", "dU/dB", "dV/dB",
     "dI/dtheta_B", "dQ/dtheta_B", "dU/dtheta_B", "dV/dtheta_B",
     "dI/dchi_B", "dQ/dchi_B", "dU/dchi_B", "dV/dchi_B"],
):
    a.plot(xgrid, resp)
    a.set_xlabel("Reduced frequency x")
    a.set_ylabel(label)
    a.grid(alpha=0.3)

fig.suptitle(f"Response to B, theta_B, chi_B at fixed h={hR_fixed_1D}, B0={B0_1D} G, delta_B={delta_B_1D} G, delta_theta_B={int(np.degrees(delta_theta_B_1D))} deg, delta_chi_B={int(np.degrees(delta_chi_B_1D))} deg")
fig.savefig(f"RF_1D_all_h{hR_fixed_1D}_B0{B0_1D}_delta_B{delta_B_1D}_delta_theta_B{int(np.degrees(delta_theta_B_1D))}_delta_chi_B{int(np.degrees(delta_chi_B_1D))}.png", dpi=300)
plt.close(fig)

fig, ax = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
for a, resp_fd, resp_grad, label in zip(
    ax.ravel(),
    [dIdB_1d, dQdB_1d, dUdB_1d, dVdB_1d],
    [dIdB_center, dQdB_center, dUdB_center, dVdB_center],
    ["dI/dB", "dQ/dB", "dU/dB", "dV/dB"],
):
    a.plot(xgrid, resp_fd, color="tab:blue", linewidth=2.0, label="finite difference")
    a.plot(xgrid, resp_grad, color="tab:orange", linestyle="--", linewidth=2.0, label="np.gradient")
    a.set_xlabel("Reduced frequency x")
    a.set_ylabel(label)
    a.grid(alpha=0.3)
    a.legend(fontsize=8)

fig.suptitle(f"Response to B at fixed h={hR_fixed_1D}, B0={B0_1D} G, delta_B={delta_B_1D} G: FD vs np.gradient")
fig.savefig(f"RF_1D_compare_B_h{hR_fixed_1D}_B0{B0_1D}_delta_B{delta_B_1D}.png", dpi=300)
plt.close(fig)

fig, ax = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
for a, resp_fd, resp_grad, label in zip(
    ax.ravel(),
    [dIdth, dQdth, dUdth, dVdth],
    [dIdth_center, dQdth_center, dUdth_center, dVdth_center],
    ["dI/dtheta_B", "dQ/dtheta_B", "dU/dtheta_B", "dV/dtheta_B"],
):
    a.plot(xgrid, resp_fd, color="tab:blue", linewidth=2.0, label="finite difference")
    a.plot(xgrid, resp_grad, color="tab:orange", linestyle="--", linewidth=2.0, label="np.gradient")
    a.set_xlabel("Reduced frequency x")
    a.set_ylabel(label)
    a.grid(alpha=0.3)
    a.legend(fontsize=8)

fig.suptitle(f"Response to theta_B at fixed h={hR_fixed_1D}, B0={B0_1D} G, delta_theta_B={int(np.degrees(delta_theta_B_1D))} deg: FD vs np.gradient")
fig.savefig(f"RF_1D_compare_theta_B_h{hR_fixed_1D}_B0{B0_1D}_delta_theta_B{int(np.degrees(delta_theta_B_1D))}deg.png", dpi=300)
plt.close(fig)

fig, ax = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
for a, resp_fd, resp_grad, label in zip(
    ax.ravel(),
    [dIdchi, dQdchi, dUdchi, dVdchi],
    [dIdchi_center, dQdchi_center, dUdchi_center, dVdchi_center],
    ["dI/dchi_B", "dQ/dchi_B", "dU/dchi_B", "dV/dchi_B"],
):
    a.plot(xgrid, resp_fd, color="tab:blue", linewidth=2.0, label="finite difference")
    a.plot(xgrid, resp_grad, color="tab:orange", linestyle="--", linewidth=2.0, label="np.gradient")
    a.set_xlabel("Reduced frequency x")
    a.set_ylabel(label)
    a.grid(alpha=0.3)
    a.legend(fontsize=8)

fig.suptitle(f"Response to chi_B at fixed h={hR_fixed_1D}, B0={B0_1D} G, delta_chi_B={int(np.degrees(delta_chi_B_1D))} deg: FD vs np.gradient")
fig.savefig(f"RF_1D_compare_chi_B_h{hR_fixed_1D}_B0{B0_1D}_delta_chi_B{int(np.degrees(delta_chi_B_1D))}deg.png", dpi=300)
plt.close(fig)

common_limit = max(
    np.max(np.abs(dIdB_1d)),
    np.max(np.abs(dUdB_1d)),
)

fig, ax = plt.subplots(1, 2, figsize=(12, 4), sharey=True)

ax[0].plot(xgrid, dIdB_1d)
ax[0].set_title("dI/dB")
ax[0].set_xlabel("Reduced frequency x")
ax[0].set_ylabel("Response")
ax[0].set_ylim(-common_limit, common_limit)
ax[0].grid(alpha=0.3)

ax[1].plot(xgrid, dUdB_1d)
ax[1].set_title("dU/dB")
ax[1].set_xlabel("Reduced frequency x")
ax[1].set_ylim(-common_limit, common_limit)
ax[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("RF_dI_dB_vs_dU_dB_same_scale.png", dpi=300)
plt.close()

print("I: FD vs gradient:",
      np.max(np.abs(dIdB_1d - dIdB_center)))

print("Q: FD vs gradient:",
      np.max(np.abs(dQdB_1d - dQdB_center)))

print("U: FD vs gradient:",
      np.max(np.abs(dUdB_1d - dUdB_center)))

print("V: FD vs gradient:",
      np.max(np.abs(dVdB_1d - dVdB_center)))

print("I and U are equal:",
      np.max(np.abs(dIdB_1d - dUdB_1d)))

fig, ax = plt.subplots(figsize=(8, 5))

ax.plot(xgrid, dIdB_1d - dUdB_1d, label="dI/dB - dU/dB")
ax.axhline(0.0, color="k", linestyle="--", linewidth=0.8)

ax.set_xlabel("Reduced frequency x")
ax.set_ylabel("Difference")
ax.set_title("Difference between dI/dB and dU/dB")
ax.grid(alpha=0.3)
ax.legend()

fig.tight_layout()
fig.savefig("RF_dI_dB_minus_dU_dB.png", dpi=300)
plt.close(fig)

geometry_tests = [
    (np.pi / 4, -np.pi / 2, "reference"),
    (np.pi / 4, -np.pi / 4, "changed azimuth"),
    (np.pi / 3, -np.pi / 2, "changed inclination"),
    (np.pi / 4, 0.0, "vertical-plane field"),
]

for theta_test, chi_test, label in geometry_tests:
    dI_test, _, dU_test, _, *_ = B_finite_difference_response_local(
        xgrid=xgrid,
        jrad=jrad_fixed,
        B0=B0_1D,
        delta_B=delta_B_1D,
        theta_B=theta_test,
        chi_B=chi_test,
        theta_obs=theta_obs,
        chi_obs=chi_obs,
        gamma_obs=gamma_obs,
        q_u_reference_mode=Q_U_REFERENCE_MODE,
    )

    print(
        label,
        "max |dI-dU| =",
        np.max(np.abs(dI_test - dU_test)),
    )

for mode in ["fixed_gamma_rotate_qu_back", "transport_gamma"]:
    dI_test, _, dU_test, _, *_ = B_finite_difference_response_local(
        xgrid=xgrid,
        jrad=jrad_fixed,
        B0=B0_1D,
        delta_B=delta_B_1D,
        theta_B=np.pi / 4,
        chi_B=-np.pi / 2,
        theta_obs=theta_obs,
        chi_obs=chi_obs,
        gamma_obs=gamma_obs,
        q_u_reference_mode=mode,
    )

    print(
        mode,
        np.max(np.abs(dI_test - dU_test)),
    )

for vary_hu, vary_vH, label in [
    (True, False, "Hanle only"),
    (False, True, "profile only"),
    (True, True, "both"),
]:
    dI_test, _, dU_test, _, *_ = B_finite_difference_response_local(
        xgrid=xgrid,
        jrad=jrad_fixed,
        B0=B0_1D,
        delta_B=delta_B_1D,
        theta_B=theta_B,
        chi_B=chi_B,
        theta_obs=theta_obs,
        chi_obs=chi_obs,
        gamma_obs=gamma_obs,
        q_u_reference_mode="fixed_gamma_rotate_qu_back",
        vary_hu_with_B=vary_hu,
        vary_vH_with_B=vary_vH,
    )

    print(
        label,
        "max |dI-dU| =",
        np.max(np.abs(dI_test - dU_test)),
    )


def stokes_from_field_angles(theta_B_value, chi_B_value):
    hu_value = hanle_parameter_exact(B0_1D, 1.0, A_ul)
    vH_value = _vH_from_B(B0_1D)
    phi_value = build_phi_table(
        xgrid,
        profile_kind=profile_kind,
        vH=vH_value,
        a_voigt=a_voigt,
    )
    state_value = prepare_magnetic_branch_state(
        jrad_fixed,
        hu_value,
        theta_B_value,
        chi_B_value,
        theta_obs,
        chi_obs,
        gamma_obs,
        Q_U_REFERENCE_MODE,
    )
    return compute_stokes_profiles(xgrid, phi_value, state_value)


pole_epsilon = np.radians(0.5)
pole_responses = {}
for pole in ("north", "south"):
    tangent_1_response, tangent_2_response = directional_response_at_field_pole(
        stokes_from_field_angles,
        B0_1D,
        pole_epsilon,
        pole=pole,
    )
    pole_responses[pole] = (tangent_1_response, tangent_2_response)

    fig, axes = plt.subplots(2, 4, figsize=(16, 7), constrained_layout=True)
    labels = ["I", "Q", "U", "V"]
    for column, label in enumerate(labels):
        axes[0, column].plot(xgrid, tangent_1_response[column])
        axes[0, column].set_title(f"D1 {label}")
        axes[0, column].set_xlabel("Reduced frequency x")
        axes[0, column].grid(alpha=0.3)

        axes[1, column].plot(xgrid, tangent_2_response[column])
        axes[1, column].set_title(f"D2 {label}")
        axes[1, column].set_xlabel("Reduced frequency x")
        axes[1, column].grid(alpha=0.3)

    fig.suptitle(
        f"Tangent-plane magnetic response at {pole} pole, "
        f"B={B0_1D} G, epsilon={np.degrees(pole_epsilon):.3g} deg"
    )
    fig.savefig(f"RF_tangent_response_{pole}_pole_B{B0_1D}.png", dpi=300)
    plt.close(fig)