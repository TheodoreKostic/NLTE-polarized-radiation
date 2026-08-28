import sys
import numpy as np

from functions_prt import wigner_D2, wigner_d2
from Radiation_fun import *
from Hanle_fun import *
from Profile_fun import *
from Chapter_13_magnetic_branch_plots import *


a_voigt = damping_parameter()
a = a_voigt


# Doppler-normalized Zeeman splitting velocity for field strength B.
def _vH_from_B(B_value):
    return 1.3996e6 * B_value / default_Delta_nu_D


# Convert a Cartesian direction vector to (theta, chi) polar/azimuthal angles.
def _angles_from_field_direction(field_direction):
    field_direction = np.asarray(field_direction, dtype=float)
    norm = np.linalg.norm(field_direction)
    if norm <= 0.0:
        raise ValueError("field_direction must be nonzero.")

    direction = field_direction / norm
    theta = np.arccos(np.clip(direction[2], -1.0, 1.0))
    chi = np.arctan2(direction[1], direction[0])
    return theta, chi


# Tangent-plane finite-difference response of Stokes params to field direction near a pole.
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


# Finite-difference derivative of Stokes I,Q,U,V with respect to field strength B.
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
    normalize=None,
    vary_hu_with_B=True,
    vary_vH_with_B=True,
):
    if delta_B <= 0.0:
        raise ValueError("delta_B must be > 0.")

    if a_value is None:
        a_value = a_voigt

    hu_ref = hanle_parameter_exact(B0, gJu, Aul)
    vH_ref = _vH_from_B(B0)

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


# Finite-difference derivative of Stokes I,Q,U,V with respect to field inclination theta_B.
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
    normalize=None,
):
    if delta_theta_B <= 0.0:
        raise ValueError("delta_theta_B must be > 0.")

    if a_value is None:
        a_value = a_voigt

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


# Finite-difference derivative of Stokes I,Q,U,V with respect to field azimuth chi_B.
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


# dStokes/dB via np.gradient over a full B sweep, for cross-check against the finite-difference version.
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


# dStokes/dtheta_B via np.gradient over a full theta_B sweep, for cross-check against the finite-difference version.
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


# dStokes/dchi_B via np.gradient over a full chi_B sweep, for cross-check against the finite-difference version.
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


# Evaluate Stokes I,Q,U,V at given field angles, falling back to __main__ globals for other params.
def stokes_from_field_angles(
    theta_B_value,
    chi_B_value,
    xgrid=None,
    jrad=None,
    B_value=None,
    theta_obs=None,
    chi_obs=None,
    gamma_obs=None,
    q_u_reference_mode="fixed_gamma_rotate_qu_back",
    gJu=1.0,
    Aul=A_ul,
    profile_kind="generalized",
    a_value=None,
):
    main_module = sys.modules.get("__main__")
    context = main_module.__dict__ if main_module is not None else globals()

    if xgrid is None:
        xgrid = context.get("xgrid")
    if jrad is None:
        jrad = context.get("jrad_fixed")
    if B_value is None:
        B_value = context.get("B0_1D")
    if theta_obs is None:
        theta_obs = context.get("theta_obs")
    if chi_obs is None:
        chi_obs = context.get("chi_obs")
    if gamma_obs is None:
        gamma_obs = context.get("gamma_obs")
    if q_u_reference_mode is None:
        q_u_reference_mode = context.get("Q_U_REFERENCE_MODE", "fixed_gamma_rotate_qu_back")

    if xgrid is None or jrad is None or B_value is None or theta_obs is None or chi_obs is None or gamma_obs is None:
        raise ValueError("Provide xgrid, jrad, B_value, theta_obs, chi_obs, gamma_obs or set the matching globals in the main script.")

    if a_value is None:
        a_value = a_voigt

    hu_value = hanle_parameter_exact(B_value, gJu, Aul)
    vH_value = _vH_from_B(B_value)
    phi_value = build_phi_table(
        xgrid,
        profile_kind=profile_kind,
        vH=vH_value,
        a_voigt=a_value,
    )
    state_value = prepare_magnetic_branch_state(
        jrad,
        hu_value,
        theta_B_value,
        chi_B_value,
        theta_obs,
        chi_obs,
        gamma_obs,
        q_u_reference_mode,
    )
    return compute_stokes_profiles(xgrid, phi_value, state_value)


# Evaluate Stokes I,Q,U,V given the magnetic field as a Cartesian vector.
def stokes_from_B_vector(
    B_vector,
    xgrid,
    jrad,
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

    B_vector = np.asarray(B_vector, dtype=float)
    B_magnitude = np.linalg.norm(B_vector)
    theta_B_value, chi_B_value = _angles_from_field_direction(B_vector)

    hu_value = hanle_parameter_exact(B_magnitude, gJu, Aul)
    vH_value = _vH_from_B(B_magnitude)
    phi_value = build_phi_table(xgrid, profile_kind=profile_kind, vH=vH_value, a_voigt=a_value)
    state_value = prepare_magnetic_branch_state(
        jrad,
        hu_value,
        theta_B_value,
        chi_B_value,
        theta_obs,
        chi_obs,
        gamma_obs,
        q_u_reference_mode,
        B_vector=B_vector,
    )
    return compute_stokes_profiles(xgrid, phi_value, state_value)


# Finite-difference derivative of Stokes I,Q,U,V with respect to each Cartesian B component (Bx, By, Bz).
def B_cartesian_finite_difference_response(
    xgrid,
    jrad,
    B_vector0,
    delta,
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
    if delta <= 0.0:
        raise ValueError("delta must be > 0.")

    B_vector0 = np.asarray(B_vector0, dtype=float)
    if np.linalg.norm(B_vector0) <= 0.0:
        raise ValueError("B_vector0 must be nonzero.")

    def stokes_at(B_vector):
        return stokes_from_B_vector(
            B_vector, xgrid, jrad, theta_obs, chi_obs, gamma_obs,
            q_u_reference_mode=q_u_reference_mode, gJu=gJu, Aul=Aul,
            profile_kind=profile_kind, a_value=a_value,
        )

    I0, Q0, U0, V0 = stokes_at(B_vector0)
    derivatives = []
    for axis in range(3):
        step = np.zeros(3)
        step[axis] = delta

        Ip, Qp, Up, Vp = stokes_at(B_vector0 + step)

        if scheme == "central":
            Im, Qm, Um, Vm = stokes_at(B_vector0 - step)
            dI = (Ip - Im) / (2.0 * delta)
            dQ = (Qp - Qm) / (2.0 * delta)
            dU = (Up - Um) / (2.0 * delta)
            dV = (Vp - Vm) / (2.0 * delta)
        elif scheme == "forward":
            dI = (Ip - I0) / delta
            dQ = (Qp - Q0) / delta
            dU = (Up - U0) / delta
            dV = (Vp - V0) / delta
        else:
            raise ValueError("scheme must be 'central' or 'forward'.")

        derivatives.append((dI, dQ, dU, dV))

    eps = 1e-300
    if normalize is None:
        return derivatives, (I0, Q0, U0, V0)

    if normalize == "I":
        denom = np.where(np.abs(I0) > eps, I0, np.nan)
        return [tuple(d / denom for d in comp) for comp in derivatives], (I0, Q0, U0, V0)

    if normalize == "self":
        dens = (
            np.where(np.abs(I0) > eps, I0, np.nan),
            np.where(np.abs(Q0) > eps, Q0, np.nan),
            np.where(np.abs(U0) > eps, U0, np.nan),
            np.where(np.abs(V0) > eps, V0, np.nan),
        )
        return [tuple(d / den for d, den in zip(comp, dens)) for comp in derivatives], (I0, Q0, U0, V0)

    raise ValueError("normalize must be None, 'I', or 'self'.")


# Chain-rule map from Cartesian B derivatives to spherical (theta_B, chi_B) derivatives.
def spherical_from_cartesian_derivatives(derivatives_cart, B_magnitude, theta_B, chi_B):
    dS_dBx, dS_dBy, dS_dBz = derivatives_cart

    dBx_dtheta = B_magnitude * np.cos(theta_B) * np.cos(chi_B)
    dBy_dtheta = B_magnitude * np.cos(theta_B) * np.sin(chi_B)
    dBz_dtheta = -B_magnitude * np.sin(theta_B)

    dBx_dchi = -B_magnitude * np.sin(theta_B) * np.sin(chi_B)
    dBy_dchi = B_magnitude * np.sin(theta_B) * np.cos(chi_B)
    dBz_dchi = 0.0

    dS_dtheta = tuple(
        dBx_dtheta * dx + dBy_dtheta * dy + dBz_dtheta * dz
        for dx, dy, dz in zip(dS_dBx, dS_dBy, dS_dBz)
    )
    dS_dchi = tuple(
        dBx_dchi * dx + dBy_dchi * dy + dBz_dchi * dz
        for dx, dy, dz in zip(dS_dBx, dS_dBy, dS_dBz)
    )
    return dS_dtheta, dS_dchi


# Chain-rule map from spherical (B, theta_B, chi_B) derivatives to Cartesian B derivatives.
def cartesian_from_spherical_derivatives(dS_dB, dS_dtheta, dS_dchi, B_magnitude, theta_B, chi_B):
    n_x = np.sin(theta_B) * np.cos(chi_B)
    n_y = np.sin(theta_B) * np.sin(chi_B)
    n_z = np.cos(theta_B)

    dtheta_dBx = np.cos(theta_B) * np.cos(chi_B) / B_magnitude
    dtheta_dBy = np.cos(theta_B) * np.sin(chi_B) / B_magnitude
    dtheta_dBz = -np.sin(theta_B) / B_magnitude

    dchi_dBx = -np.sin(chi_B) / (B_magnitude * np.sin(theta_B))
    dchi_dBy = np.cos(chi_B) / (B_magnitude * np.sin(theta_B))
    dchi_dBz = 0.0

    dS_dBx = tuple(db * n_x + dth * dtheta_dBx + dch * dchi_dBx for db, dth, dch in zip(dS_dB, dS_dtheta, dS_dchi))
    dS_dBy = tuple(db * n_y + dth * dtheta_dBy + dch * dchi_dBy for db, dth, dch in zip(dS_dB, dS_dtheta, dS_dchi))
    dS_dBz = tuple(db * n_z + dth * dtheta_dBz + dch * dchi_dBz for db, dth, dch in zip(dS_dB, dS_dtheta, dS_dchi))

    return dS_dBx, dS_dBy, dS_dBz

# Compare direct Cartesian finite differences with the full spherical-to-Cartesian chain rule.
# Returns a dict with:
#   - "direct": direct derivatives as (dS/dBx, dS/dBy, dS/dBz)
#   - "chain_rule": reconstructed derivatives via cartesian_from_spherical_derivatives
#   - "max_abs_diff": per-axis maximum absolute difference for I, Q, U, V
#   - "theta_B", "chi_B", "B_magnitude": geometry used for the conversion
def compare_cartesian_derivative_methods(
    xgrid,
    jrad,
    B_vector0,
    delta_B,
    theta_obs,
    chi_obs,
    gamma_obs,
    q_u_reference_mode="fixed_gamma_rotate_qu_back",
    profile_kind="generalized",
    delta_theta_B=None,
    delta_chi_B=None,
    a_value=None,
    gJu=1.0,
    Aul=A_ul,
    scheme="central",
):
    """Compare direct Cartesian finite differences with the full spherical-to-Cartesian chain rule.

    Returns a dict with:
      - "direct": direct derivatives as (dS/dBx, dS/dBy, dS/dBz)
      - "chain_rule": reconstructed derivatives via cartesian_from_spherical_derivatives
      - "max_abs_diff": per-axis maximum absolute difference for I, Q, U, V
      - "theta_B", "chi_B", "B_magnitude": geometry used for the conversion
    """
    B_vector0 = np.asarray(B_vector0, dtype=float)
    if np.linalg.norm(B_vector0) <= 0.0:
        raise ValueError("B_vector0 must be nonzero.")

    B_magnitude = np.linalg.norm(B_vector0)
    theta_B, chi_B = _angles_from_field_direction(B_vector0)

    if delta_theta_B is None:
        delta_theta_B = np.radians(5.0)
    if delta_chi_B is None:
        delta_chi_B = np.radians(5.0)

    direct_derivatives, _ = B_cartesian_finite_difference_response(
        xgrid=xgrid,
        jrad=jrad,
        B_vector0=B_vector0,
        delta=delta_B,
        theta_obs=theta_obs,
        chi_obs=chi_obs,
        gamma_obs=gamma_obs,
        q_u_reference_mode=q_u_reference_mode,
        gJu=gJu,
        Aul=Aul,
        profile_kind=profile_kind,
        a_value=a_value,
        scheme=scheme,
        normalize=None,
    )

    dIdB, dQdB, dUdB, dVdB, *_ = B_finite_difference_response_local(
        xgrid=xgrid,
        jrad=jrad,
        B0=B_magnitude,
        delta_B=delta_B,
        theta_B=theta_B,
        chi_B=chi_B,
        theta_obs=theta_obs,
        chi_obs=chi_obs,
        gamma_obs=gamma_obs,
        q_u_reference_mode=q_u_reference_mode,
        gJu=gJu,
        Aul=Aul,
        profile_kind=profile_kind,
        a_value=a_value,
        scheme=scheme,
        normalize=None,
    )

    dIdtheta, dQdtheta, dUdtheta, dVdtheta, *_ = theta_B_finite_difference_response_local(
        xgrid=xgrid,
        jrad=jrad,
        B_value=B_magnitude,
        theta_B0=theta_B,
        delta_theta_B=delta_theta_B,
        chi_B=chi_B,
        theta_obs=theta_obs,
        chi_obs=chi_obs,
        gamma_obs=gamma_obs,
        q_u_reference_mode=q_u_reference_mode,
        gJu=gJu,
        Aul=Aul,
        profile_kind=profile_kind,
        a_value=a_value,
        scheme=scheme,
        normalize=None,
    )

    dIdchi, dQdchi, dUdchi, dVdchi, *_ = chi_B_finite_difference_response_local(
        xgrid=xgrid,
        jrad=jrad,
        B_value=B_magnitude,
        theta_B=theta_B,
        chi_B0=chi_B,
        delta_chi_B=delta_chi_B,
        theta_obs=theta_obs,
        chi_obs=chi_obs,
        gamma_obs=gamma_obs,
        q_u_reference_mode=q_u_reference_mode,
        gJu=gJu,
        Aul=Aul,
        profile_kind=profile_kind,
        a_value=a_value,
        scheme=scheme,
        normalize=None,
    )

    dS_dB = (dIdB, dQdB, dUdB, dVdB)
    dS_dtheta = (dIdtheta, dQdtheta, dUdtheta, dVdtheta)
    dS_dchi = (dIdchi, dQdchi, dUdchi, dVdchi)

    chain_rule = cartesian_from_spherical_derivatives(
        dS_dB,
        dS_dtheta,
        dS_dchi,
        B_magnitude,
        theta_B,
        chi_B,
    )

    max_abs_diff = {}
    for axis_name, axis_idx in zip(("Bx", "By", "Bz"), range(3)):
        max_abs_diff[axis_name] = {}
        for stokes_idx, stokes_name in enumerate(("I", "Q", "U", "V")):
            max_abs_diff[axis_name][stokes_name] = np.max(
                np.abs(np.asarray(chain_rule[axis_idx][stokes_idx]) - np.asarray(direct_derivatives[axis_idx][stokes_idx]))
            )

    return {
        "direct": direct_derivatives,
        "chain_rule": chain_rule,
        "theta_B": theta_B,
        "chi_B": chi_B,
        "B_magnitude": B_magnitude,
        "max_abs_diff": max_abs_diff,
    }


# All J^K_Q components carried through the magnetic-branch pipeline: (0,0) plus the five K=2 alignment terms.
J_KQ_KEYS = [(0, 0), (2, -2), (2, -1), (2, 0), (2, 1), (2, 2)]


# Evaluate Stokes I,Q,U,V for an arbitrary user-supplied radiation tensor dict.
def stokes_from_jrad(
    jrad,
    xgrid,
    hu,
    theta_B,
    chi_B,
    theta_obs,
    chi_obs,
    gamma_obs,
    q_u_reference_mode="fixed_gamma_rotate_qu_back",
    profile_kind="generalized",
    a_value=None,
    vH=None,
):
    if a_value is None:
        a_value = a_voigt
    if vH is None:
        raise ValueError("vH must be provided (it is not derivable from hu alone).")

    phi_value = build_phi_table(xgrid, profile_kind=profile_kind, vH=vH, a_voigt=a_value)
    state_value = prepare_magnetic_branch_state(
        jrad, hu, theta_B, chi_B, theta_obs, chi_obs, gamma_obs, q_u_reference_mode,
    )
    return compute_stokes_profiles(xgrid, phi_value, state_value)


def J_component_finite_difference_response(
    xgrid,
    jrad_base,
    key,
    part,
    delta,
    hu,
    vH,
    theta_B,
    chi_B,
    theta_obs,
    chi_obs,
    gamma_obs,
    q_u_reference_mode="fixed_gamma_rotate_qu_back",
    profile_kind="generalized",
    a_value=None,
    scheme="central",
    normalize=None,
):
    """Response of I,Q,U,V to a perturbation of Re/Im of a single J^K_Q component (key = (K, Q))."""
    if delta <= 0.0:
        raise ValueError("delta must be > 0.")
    if key not in jrad_base:
        raise ValueError(f"key {key} not present in jrad_base.")
    if part not in ("real", "imag"):
        raise ValueError("part must be 'real' or 'imag'.")
    if key == (0, 0) and part == "imag":
        raise ValueError("J^0_0 is real; perturbing its imaginary part is not physical.")

    def perturb(amount):
        jrad = dict(jrad_base)
        jrad[key] = jrad[key] + (amount if part == "real" else 1j * amount)
        return jrad

    def stokes_for(jrad):
        return stokes_from_jrad(
            jrad, xgrid, hu, theta_B, chi_B, theta_obs, chi_obs, gamma_obs,
            q_u_reference_mode=q_u_reference_mode, profile_kind=profile_kind,
            a_value=a_value, vH=vH,
        )

    Ip, Qp, Up, Vp = stokes_for(perturb(delta))

    if scheme == "central":
        Im, Qm, Um, Vm = stokes_for(perturb(-delta))
        dI = (Ip - Im) / (2.0 * delta)
        dQ = (Qp - Qm) / (2.0 * delta)
        dU = (Up - Um) / (2.0 * delta)
        dV = (Vp - Vm) / (2.0 * delta)
        I0, Q0, U0, V0 = stokes_for(jrad_base)
    elif scheme == "forward":
        I0, Q0, U0, V0 = stokes_for(jrad_base)
        dI = (Ip - I0) / delta
        dQ = (Qp - Q0) / delta
        dU = (Up - U0) / delta
        dV = (Vp - V0) / delta
    else:
        raise ValueError("scheme must be 'central' or 'forward'.")

    eps = 1e-300
    if normalize is None:
        return dI, dQ, dU, dV, I0, Q0, U0, V0

    if normalize == "I":
        denom = np.where(np.abs(I0) > eps, I0, np.nan)
        return dI / denom, dQ / denom, dU / denom, dV / denom, I0, Q0, U0, V0

    if normalize == "self":
        denI = np.where(np.abs(I0) > eps, I0, np.nan)
        denQ = np.where(np.abs(Q0) > eps, Q0, np.nan)
        denU = np.where(np.abs(U0) > eps, U0, np.nan)
        denV = np.where(np.abs(V0) > eps, V0, np.nan)
        return dI / denI, dQ / denQ, dU / denU, dV / denV, I0, Q0, U0, V0

    raise ValueError("normalize must be None, 'I', or 'self'.")


# Full finite-difference Jacobian of I,Q,U,V with respect to Re/Im of every J^K_Q component.
def J_jacobian_finite_difference(
    xgrid,
    jrad_base,
    delta,
    hu,
    vH,
    theta_B,
    chi_B,
    theta_obs,
    chi_obs,
    gamma_obs,
    q_u_reference_mode="fixed_gamma_rotate_qu_back",
    profile_kind="generalized",
    a_value=None,
    scheme="central",
    normalize=None,
    keys=None,
):
    """Full Jacobian d(I,Q,U,V)/d(Re,Im) for every requested J^K_Q component.

    Returns a dict {(K, Q, part): (dI, dQ, dU, dV)}.
    """
    if keys is None:
        keys = J_KQ_KEYS

    jacobian = {}
    for key in keys:
        parts = ("real",) if key == (0, 0) else ("real", "imag")
        for part in parts:
            dI, dQ, dU, dV, *_ = J_component_finite_difference_response(
                xgrid, jrad_base, key, part, delta, hu, vH,
                theta_B, chi_B, theta_obs, chi_obs, gamma_obs,
                q_u_reference_mode=q_u_reference_mode, profile_kind=profile_kind,
                a_value=a_value, scheme=scheme, normalize=normalize,
            )
            jacobian[(key[0], key[1], part)] = (dI, dQ, dU, dV)

    return jacobian
