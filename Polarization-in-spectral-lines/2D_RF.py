import numpy as np
import sys
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

script_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(script_dir)

from functions_prt import wigner_D2, wigner_d2
from Radiation_fun import *
from Hanle_fun import *
from Profile_fun import *
from Chapter_13_magnetic_branch_plots import *
from Derivates import (
    B_cartesian_finite_difference_response,
    B_finite_difference_response_local,
    J_component_finite_difference_response,
    J_jacobian_finite_difference,
    J_KQ_KEYS,
    cartesian_from_spherical_derivatives,
    chi_B_finite_difference_response_local,
    compare_cartesian_derivative_methods,
    directional_response_at_field_pole,
    J_jacobian_vs_B,
    response_vs_B_for_angle_perturbation,
    response_vs_B_gradient,
    response_vs_chi_B_gradient,
    response_vs_theta_B_gradient,
    spherical_from_cartesian_derivatives,
    stokes_from_B_vector,
    stokes_from_field_angles,
    theta_B_finite_difference_response_local,
)

# 2D Response Functions for the case angle
# delta = 0, i.e, theta_obs = 90.
# On x-axis we have reduced wavelength x
# On y-axis we have the magnetic field strength B
# Colorbar represents the magnitude of the response function

hR_2D = 0.073
jrad_2D_test = radiation_tensor(hR_2D)

xgrid = np.linspace(-5.0, 5.0, 200)
theta_B = np.pi/2 
chi_B = 0.0
theta_obs = np.pi/2
chi_obs = 0.0
gamma_obs = np.pi/2

profile_kind = "generalized"
Q_U_REFERENCE_MODE = "fixed_gamma_rotate_qu_back"

B_array = np.linspace(1.0, 50.0, 60)
delta_B_2D = 0.2
delta_theta_B_2D = np.radians(2.0)
delta_chi_B_2D = np.radians(2.0)
delta_J_2D = 1e-4

STOKES_LABELS = ["I", "Q", "U", "V"]

# ASCII tag for filenames; human-readable Greek/"=" form for suptitles
geometry_tag = f"h_{hR_2D}_chi_B_{np.degrees(chi_B):.0f}_theta_B_{np.degrees(theta_B):.0f}"
geometry_title = f"h = {hR_2D}, χ_B = {np.degrees(chi_B):.0f}°, θ_B = {np.degrees(theta_B):.0f}°"


def plot_2d_response_grid(maps, filename_tag, title):
    fig, ax = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for a, resp, label in zip(ax.ravel(), maps, STOKES_LABELS):
        vmax = np.max(np.abs(resp))
        vmax = vmax if vmax > 0 else 1.0
        mesh = a.pcolormesh(xgrid, B_array, resp, shading="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        fig.colorbar(mesh, ax=a, label=f"d{label}")
        a.set_xlabel("Reduced frequency x")
        a.set_ylabel("B (G)")
        a.set_title(label)
    fig.suptitle(title)
    fig.savefig(f"RF_2D_{filename_tag}_{geometry_tag}.png", dpi=300)
    plt.close(fig)


# --- Response to B itself ---
dIdB_2d, dQdB_2d, dUdB_2d, dVdB_2d, *_ = response_vs_B_gradient(
    xgrid, jrad_2D_test, B_array, theta_B, chi_B, theta_obs, chi_obs, gamma_obs,
    q_u_reference_mode=Q_U_REFERENCE_MODE, profile_kind=profile_kind,
)
plot_2d_response_grid(
    [dIdB_2d, dQdB_2d, dUdB_2d, dVdB_2d],
    "B",
    f"Response to B, {geometry_title}",
)

# --- Response to theta_B, as a function of background B ---
dIdth_2d, dQdth_2d, dUdth_2d, dVdth_2d, *_ = response_vs_B_for_angle_perturbation(
    theta_B_finite_difference_response_local, xgrid, jrad_2D_test, B_array,
    theta_B0=theta_B, delta_theta_B=delta_theta_B_2D, chi_B=chi_B,
    theta_obs=theta_obs, chi_obs=chi_obs, gamma_obs=gamma_obs,
    q_u_reference_mode=Q_U_REFERENCE_MODE, profile_kind=profile_kind,
)
plot_2d_response_grid(
    [dIdth_2d, dQdth_2d, dUdth_2d, dVdth_2d],
    "theta_B",
    f"Response to θ_B, {geometry_title}, Δθ_B = {np.degrees(delta_theta_B_2D):.0f}°",
)

# --- Response to chi_B, as a function of background B ---
dIdchi_2d, dQdchi_2d, dUdchi_2d, dVdchi_2d, *_ = response_vs_B_for_angle_perturbation(
    chi_B_finite_difference_response_local, xgrid, jrad_2D_test, B_array,
    theta_B=theta_B, chi_B0=chi_B, delta_chi_B=delta_chi_B_2D,
    theta_obs=theta_obs, chi_obs=chi_obs, gamma_obs=gamma_obs,
    q_u_reference_mode=Q_U_REFERENCE_MODE, profile_kind=profile_kind,
)
plot_2d_response_grid(
    [dIdchi_2d, dQdchi_2d, dUdchi_2d, dVdchi_2d],
    "chi_B",
    f"Response to χ_B, {geometry_title}, Δχ_B = {np.degrees(delta_chi_B_2D):.0f}°",
)

# --- Response to J^K_Q components, as a function of background B ---
j_maps_vs_B = J_jacobian_vs_B(
    xgrid, jrad_2D_test, delta_J_2D, B_array, theta_B, chi_B,
    theta_obs, chi_obs, gamma_obs,
    q_u_reference_mode=Q_U_REFERENCE_MODE, profile_kind=profile_kind,
)
for (K, Q, part), stokes_maps in j_maps_vs_B.items():
    part_symbol = "Re" if part == "real" else "Im"
    plot_2d_response_grid(
        [stokes_maps["I"], stokes_maps["Q"], stokes_maps["U"], stokes_maps["V"]],
        f"J{K}_{Q}_{part_symbol}",
        rf"Response to {part_symbol}($J^{K}_{{{Q}}}$), {geometry_title}, $\Delta${part_symbol}($J^{K}_{{{Q}}}$) = {delta_J_2D:.0e}",
    )

print("Finished calculating and plotting 2D response function maps.")