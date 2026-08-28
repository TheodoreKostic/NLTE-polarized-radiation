import numpy as np
import sys
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

#script_dir = os.path.abspath("/home/Code/NLTE-polarized-radiation")
#script_dir = os.path.abspath("/home/teodor/Documents/Codes/NLTE-polarized-radiation")
script_dir = os.path.abspath("/home/mistflow/Documents/Doktorat/NLTE-polarized-radiation")
sys.path.append(script_dir)

from functions_prt import wigner_D2, wigner_d2
from Radiation_fun import *
from Hanle_fun import *
from Profile_fun import *
from Chapter_13_magnetic_branch_plots import *
from Derivates import (
    B_cartesian_finite_difference_response,
    B_finite_difference_response_local,
    cartesian_from_spherical_derivatives,
    chi_B_finite_difference_response_local,
    compare_cartesian_derivative_methods,
    directional_response_at_field_pole,
    response_vs_B_gradient,
    response_vs_chi_B_gradient,
    response_vs_theta_B_gradient,
    spherical_from_cartesian_derivatives,
    stokes_from_B_vector,
    stokes_from_field_angles,
    theta_B_finite_difference_response_local,
)

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
theta_B = np.pi/4 # np.pi/3
chi_B = -np.pi/2 # np.pi/6
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


B_vector0 = B0_1D * np.array([
    np.sin(theta_B) * np.cos(chi_B),
    np.sin(theta_B) * np.sin(chi_B),
    np.cos(theta_B),
])

derivatives_cart, (I0c, Q0c, U0c, V0c) = B_cartesian_finite_difference_response(
    xgrid=xgrid,
    jrad=jrad_fixed,
    B_vector0=B_vector0,
    delta=delta_B_1D,
    theta_obs=theta_obs,
    chi_obs=chi_obs,
    gamma_obs=gamma_obs,
    q_u_reference_mode=Q_U_REFERENCE_MODE,
    profile_kind=profile_kind,
    scheme="central",
    normalize=None,
)

axis_labels = ["Bx", "By", "Bz"]
fig, ax = plt.subplots(3, 4, figsize=(18, 10), constrained_layout=True)
for row, (deriv, axis_label) in enumerate(zip(derivatives_cart, axis_labels)):
    for col, (resp, stokes_label) in enumerate(zip(deriv, ["I", "Q", "U", "V"])):
        ax[row, col].plot(xgrid, resp)
        ax[row, col].set_title(f"d{stokes_label}/d{axis_label}")
        ax[row, col].set_xlabel("Reduced frequency x")
        ax[row, col].grid(alpha=0.3)

fig.suptitle(f"Cartesian response at theta_B={np.degrees(theta_B):.1f} deg, chi_B={np.degrees(chi_B):.1f} deg, B0={B0_1D} G")
fig.savefig(f"RF_cartesian_response_B0{B0_1D}_chiB{np.degrees(chi_B):.1f}_thetaB{np.degrees(theta_B):.1f}.png", dpi=300)
plt.close(fig)

dI_chi, dQ_chi, dU_chi, dV_chi, *_ = chi_B_finite_difference_response_local(
    xgrid=xgrid, jrad=jrad_fixed, B_value=B0_1D,
    theta_B=theta_B, chi_B0=chi_B, delta_chi_B=delta_chi_B_1D,
    theta_obs=theta_obs, chi_obs=chi_obs, gamma_obs=gamma_obs,
    q_u_reference_mode=Q_U_REFERENCE_MODE, profile_kind=profile_kind,
)
print("max |dI/dchi - dQ/dchi| =", np.max(np.abs(dI_chi - dQ_chi)))

dIdB_radial, dQdB_radial, dUdB_radial, dVdB_radial, *_ = B_finite_difference_response_local(
    xgrid=xgrid, jrad=jrad_fixed, B0=B0_1D, delta_B=delta_B_1D,
    theta_B=theta_B, chi_B=chi_B, theta_obs=theta_obs, chi_obs=chi_obs,
    gamma_obs=gamma_obs, q_u_reference_mode=Q_U_REFERENCE_MODE, profile_kind=profile_kind,
)
n_x = np.sin(theta_B) * np.cos(chi_B)
approx_dIdBx = dIdB_radial * n_x   # the shortcut

B_vector0 = B0_1D * np.array([np.sin(theta_B)*np.cos(chi_B), np.sin(theta_B)*np.sin(chi_B), np.cos(theta_B)])
derivatives_cart, _ = B_cartesian_finite_difference_response(
    xgrid=xgrid, jrad=jrad_fixed, B_vector0=B_vector0, delta=delta_B_1D,
    theta_obs=theta_obs, chi_obs=chi_obs, gamma_obs=gamma_obs,
    q_u_reference_mode=Q_U_REFERENCE_MODE, profile_kind=profile_kind,
)
true_dIdBx = derivatives_cart[0][0]  # dS/dBx, I-component
true_dIdBy = derivatives_cart[1][0]  # dS/dBy, I-component
true_dIdBz = derivatives_cart[2][0]  # dS/dBz, I-component

print("max |shortcut - true| =", np.max(np.abs(approx_dIdBx - true_dIdBx)))

comparison = compare_cartesian_derivative_methods(
    xgrid=xgrid,
    jrad=jrad_fixed,
    B_vector0=B_vector0,
    delta_B=delta_B_1D,
    theta_obs=theta_obs,
    chi_obs=chi_obs,
    gamma_obs=gamma_obs,
    q_u_reference_mode=Q_U_REFERENCE_MODE,
    profile_kind=profile_kind,
    delta_theta_B=delta_theta_B_1D,
    delta_chi_B=delta_chi_B_1D,
    a_value=None,
    gJu=1.0,
    Aul=A_ul,
    scheme="central",
)

axis_labels = ["Bx", "By", "Bz"]
stokes_labels = ["I", "Q", "U", "V"]
fig, ax = plt.subplots(3, 4, figsize=(18, 10), constrained_layout=True)
for row, (direct_axis, chain_axis, axis_label) in enumerate(
    zip(comparison["direct"], comparison["chain_rule"], axis_labels)
):
    for column, (direct_response, chain_response, stokes_label) in enumerate(
        zip(direct_axis, chain_axis, stokes_labels)
    ):
        ax[row, column].plot(
            xgrid,
            direct_response,
            color="tab:blue",
            linewidth=2.0,
            label="direct Cartesian",
        )
        ax[row, column].plot(
            xgrid,
            chain_response,
            color="tab:orange",
            linestyle=":",
            linewidth=2.0,
            label="full chain rule",
        )
        ax[row, column].set_title(f"d{stokes_label}/d{axis_label}")
        ax[row, column].set_xlabel("Reduced frequency x")
        ax[row, column].grid(alpha=0.3)
        ax[row, column].legend(fontsize=8)

fig.suptitle(
    f"Direct Cartesian vs full chain-rule response, B0={B0_1D} G, "
    f"chiB={np.degrees(chi_B):.1f} deg, thetaB={np.degrees(theta_B):.1f} deg"
)
fig.savefig(
    f"RF_1D_comparison_{B0_1D}_{np.degrees(chi_B):.1f}_{np.degrees(theta_B):.1f}.png",
    dpi=300,
)
plt.close(fig)

print("full chain-rule vs direct Cartesian max absolute differences:")
for axis_name, axis_diffs in comparison["max_abs_diff"].items():
    print(axis_name, {stokes_name: float(value) for stokes_name, value in axis_diffs.items()})


# Response functions computed by perturbing Bx, By, and Bz independently
# while evaluating Stokes quantities through the Cartesian-vector path.
cartesian_path_derivatives, _ = B_cartesian_finite_difference_response(
    xgrid=xgrid,
    jrad=jrad_fixed,
    B_vector0=B_vector0,
    delta=delta_B_1D,
    theta_obs=theta_obs,
    chi_obs=chi_obs,
    gamma_obs=gamma_obs,
    q_u_reference_mode=Q_U_REFERENCE_MODE,
    profile_kind=profile_kind,
    scheme="central",
    normalize=None,
)

axis_labels = ["Bx", "By", "Bz"]
stokes_labels = ["I", "Q", "U", "V"]
fig, ax = plt.subplots(3, 4, figsize=(18, 10), constrained_layout=True)
for row, (axis_derivatives, axis_label) in enumerate(zip(cartesian_path_derivatives, axis_labels)):
    for column, (response, stokes_label) in enumerate(zip(axis_derivatives, stokes_labels)):
        ax[row, column].plot(xgrid, response, color="tab:blue", linewidth=2.0)
        ax[row, column].set_title(f"d{stokes_label}/d{axis_label}")
        ax[row, column].set_xlabel("Reduced frequency x")
        ax[row, column].set_ylabel("Response")
        ax[row, column].grid(alpha=0.3)

fig.suptitle(
    f"Cartesian-vector response functions, B0={B0_1D} G, "
    f"chiB={np.degrees(chi_B):.1f} deg, thetaB={np.degrees(theta_B):.1f} deg"
)
fig.savefig(
    f"RF_1D_cartesian_path_response_{B0_1D}_{np.degrees(chi_B):.1f}_{np.degrees(theta_B):.1f}.png",
    dpi=300,
)
plt.close(fig)

