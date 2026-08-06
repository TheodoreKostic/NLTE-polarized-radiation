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

a_voigt = damping_parameter()
a = a_voigt

def _trapz(y, x):
    if hasattr(np, "trapezoid"):
        return np.trapezoid(y, x)
    return np.trapz(y, x)


B = 5.69 # G
mu_B = 9.274 * 10e-21
h = 6.626 * 10e-27
#vH = mu_B * B / h / default_Delta_nu_D
Jrad_0 = radiation_tensor(hR=0.073)
#Hu = 1.0 # depends on B
#vH = 5.0
Hu = hanle_parameter_exact(5.69, 1.0, A_ul)
vH = 1.3996e6 * B / default_Delta_nu_D

print("vH = ", vH)
print("a_voigt = ", a_voigt)

xgrid = np.linspace(-5,5,101)

I_prof = np.zeros_like(xgrid)
Q_prof = np.zeros_like(xgrid)
U_prof = np.zeros_like(xgrid)
V_prof = np.zeros_like(xgrid)

theta_B = np.pi/4 # np.pi/4
chi_B = -np.pi/2 # -np.pi/2
theta_obs = np.pi/2
chi_obs = 0.0
gamma_obs = np.pi/2

# ---------------------------------------------------------
# Frame toggle for Eq. (13.20) contraction
# ---------------------------------------------------------
# False: current vertical-frame path (full Hanle operator)
# True : LL04-style magnetic-frame contraction (single frame)
USE_LL04_MAG_FRAME_BRANCH = True

# Q/U reference handling when using magnetic-frame branch:
# "transport_gamma"            -> transport +Q axis and use transported gamma directly
# "fixed_gamma_rotate_qu_back" -> keep original gamma in contraction, rotate Q/U back after
Q_U_REFERENCE_MODE = "fixed_gamma_rotate_qu_back"


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


Jarr_base = Jrad_to_array(Jrad_0)
J00_base = Jrad_0[(0,0)]
qu_back_rotation = 0.0

if USE_LL04_MAG_FRAME_BRANCH:
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
else:
    rho2_base = apply_hanle(Jarr_base, Hu, theta_B, chi_B)
    qu_back_rotation = 0.0
    print("Frame branch: vertical-frame full Hanle operator")

Phi = {}

for K in [0,1,2]:
    for Kp in [0,1,2]:
        for Q in [-2,-1,0,1,2]:

            Phi[(K,Kp,Q)] = Phi_generalized(
                xgrid,
                K,
                Kp,
                Q,
                vH,
                a_voigt
            )

print("2,1,0 x =-1, =  ", np.max(np.abs(Phi[(2,1,0)])))

# ----------------------------------------------------------
# Plot all generalized profiles
# ----------------------------------------------------------

q_colors = {
    -2: "tab:purple",
    -1: "tab:red",
     0: "tab:green",
     1: "tab:orange",
     2: "tab:blue",
}

q_styles = {
    -2: "-",
    -1: "--",
    0: "-.",
    1: ":",
    2: (0,(5,2)),
}

fig, axes = plt.subplots(
    6,
    2,
    figsize=(15, 20),
    constrained_layout=True
)

pairs = [
    (0,0,[0]),
    (0,1,[0]),
    (0,2,[0]),
    (2,0,[0]),
    (2,1,[-2,-1,0,1,2]),
    (2,2,[-2,-1,0,1,2]),
]

for row, (K,Kp,Qlist) in enumerate(pairs):

    # -----------------------------
    # Real part
    # -----------------------------

    ax = axes[row,0]

    for Q in Qlist:

        Phi, components = Phi_generalized(
            xgrid,
            K,
            Kp,
            Q,
            vH,
            a_voigt,
            return_pairs=True
        )

        color = q_colors[Q]
        style = q_styles[Q]

        # reference Voigt profile
        if Q == Qlist[0]:
            ax.plot(
                xgrid,
                np.real(phi_complex(xgrid, a_voigt)),
                "k--",
                alpha=0.3,
                linewidth=1,
                label="Voigt"
            )

        # individual Mu,Mu' contributions
        for Mu in (-1,0,1):
            for Mup in (-1,0,1):

                ax.plot(
                    xgrid,
                    np.real(components[Mu+1, Mup+1]),
                    color=color,
                    linestyle=style,
                    linewidth=0.8,
                    alpha=0.6
                )

        # total generalized profile
        ax.plot(
            xgrid,
            np.real(Phi),
            color=color,
            linewidth=2.0,
            label=rf"Total $Q={Q}$"
        )

    ax.set_title(rf"$K={K},\,K'={Kp}$")
    ax.set_ylabel(rf"$K={K},\,K'={Kp}$")
    ax.grid(alpha=0.3)

    # Legend only once
    if row == 0:
        ax.legend(
            loc="center left",
            bbox_to_anchor=(1.02,0.5),
            fontsize=8,
            frameon=True
        )

    # -----------------------------
    # Imaginary part
    # -----------------------------

    ax = axes[row,1]

    for Q in Qlist:

        Phi, components = Phi_generalized(
            xgrid,
            K,
            Kp,
            Q,
            vH,
            a_voigt,
            return_pairs=True
        )

        color = q_colors[Q]

        for Mu in (-1,0,1):
            for Mup in (-1,0,1):

                ax.plot(
                    xgrid,
                    np.imag(components[Mu+1, Mup+1]),
                    color=color,
                    linestyle=style,
                    linewidth=0.8,
                    alpha=0.6
                )

        ax.plot(
            xgrid,
            np.imag(Phi),
            color=color,
            linewidth=2.0,
            label=rf"Total $Q={Q}$"
        )

    ax.grid(alpha=0.3)
    ax.legend(
    fontsize=5,
    ncol=2,
    loc="upper right",
    framealpha=0.9
)

axes[-1,0].set_xlabel("Reduced frequency $x$")
axes[-1,1].set_xlabel("Reduced frequency $x$")

fig.suptitle("Generalized profiles", fontsize=16)

plt.savefig(
    "Generalized_profiles_colored.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()

# ---------------------------------------------------------
# Test symmetry relations
# ---------------------------------------------------------

Phi = {}

for K in [0,2]:
    for Kp in [0,1,2]:
        for Q in [-2,-1,0,1,2]:

            Phi[(K,Kp,Q)] = Phi_generalized(
                xgrid,
                K,
                Kp,
                Q,
                vH,
                a_voigt
            )


print("\n==============================")
print("SYMMETRY TESTS")
print("==============================")

for K in [0,2]:

    for Kp in [0,1,2]:

        print(f"\nK={K}, K'={Kp}")

        for Q in [1,2]:

            Pplus = Phi[(K,Kp,Q)]
            Pminus = Phi[(K,Kp,-Q)]

            lhs = Pminus
            rhs = (-1)**Q * np.conj(Pplus)

            err = np.max(np.abs(lhs-rhs))

            print(
                f"Q={Q}: "
                f"max |Φ(-Q)-(-1)^Q Φ(Q)*| = {err:.3e}"
            )

# ---------------------------------------------------------
# Compare normalized shapes
# ---------------------------------------------------------

print("\n==============================")
print("NORMALIZED SHAPE TEST")
print("==============================")

for K in [0,2]:

    for Kp in [0,1,2]:

        print(f"\nK={K}, K'={Kp}")

        ref = None

        for Q in [-2,-1,0,1,2]:

            prof = Phi[(K,Kp,Q)]

            amp = np.max(np.abs(np.real(prof)))

            if amp < 1e-12:
                continue

            norm = np.real(prof)/amp

            if ref is None:

                ref = norm

                continue

            diff = np.max(np.abs(norm-ref))

            print(
                f"Q={Q}: normalized shape difference = {diff:.3e}"
            )

# ---------------------------------------------------------
# Minimal diagnostics for Fig. 13.6 symmetry sensitivity
# ---------------------------------------------------------

RUN_DIAG = True
OLD_DEBUG = False
DIAG_X_POINTS = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])


def old_debug_print(*args, **kwargs):
    if OLD_DEBUG:
        print(*args, **kwargs)


def complex_phase_deg(z):
    if np.abs(z) < 1e-300:
        return np.nan
    return np.degrees(np.angle(z))


def diag_is_sample_x(x):
    return np.any(np.isclose(x, DIAG_X_POINTS, atol=1e-12))


def diag_print_profile_family_mismatch():
    # Checklist #3: strict profile-family isolation
    gen_p1 = Phi_generalized(xgrid, 2, 1, 1, vH, a_voigt)
    gen_m1 = Phi_generalized(xgrid, 2, 1, -1, vH, a_voigt)

    app_p1 = Phi_appendix(xgrid, 2, 1, 1, vH, a_voigt)
    app_m1 = Phi_appendix(xgrid, 2, 1, -1, vH, a_voigt)

    print("\n[DIAG #3] Profile family mismatch (K=2, Kp=1)")
    print("max |Re(gen_p1 - app_p1)| =", np.max(np.abs(np.real(gen_p1 - app_p1))))
    print("max |Im(gen_p1 - app_p1)| =", np.max(np.abs(np.imag(gen_p1 - app_p1))))
    print("max |Re(gen_m1 - app_m1)| =", np.max(np.abs(np.real(gen_m1 - app_m1))))
    print("max |Im(gen_m1 - app_m1)| =", np.max(np.abs(np.imag(gen_m1 - app_m1))))


def diag_v21_variant_at_ix(ix, rho2, use_conj_minusQ=True, flip_phase=False):
    # Checklist #5: micro-perturbation sensitivity
    v = 0.0 + 0.0j

    for Q in [-1, 0, 1]:
        phase = (-1)**Q
        if flip_phase:
            phase *= -1.0

        if use_conj_minusQ:
            rhoQ = np.conj(rho2[idx(-Q)])
        else:
            rhoQ = rho2[idx(Q)]

        phiQ = Phi[(2,1,Q)][ix]
        tQ = T(3,1,Q,theta_obs,chi_obs,gamma_obs)
        v += phase * phiQ * tQ * rhoQ

    return v


if RUN_DIAG:
    diag_print_profile_family_mismatch()

V21_profile = np.zeros_like(xgrid)
V21_2_profile = np.zeros_like(xgrid)
V21_0_profile = np.zeros_like(xgrid)
V21_m2_profile = np.zeros_like(xgrid)
V21_m1_profile = np.zeros_like(xgrid)
V01_profile = np.zeros_like(xgrid)
for ix, x in enumerate(xgrid):
    J00 = J00_base
    rho2 = rho2_base
    #print("\nFrequency x = ", x)
    #print("rho2 real:", np.real(rho2))
    #print("rho2 imag:", np.imag(rho2))
    epsI = 0.0+0j
    epsQ = 0.0+0j
    epsU = 0.0+0j
    epsV = 0.0+0j
    # K=0 blocks
    #Phi00 = Phi_generalized(np.array([x]), K=0, Kp=0, Q=0, vH=vH, a=a_voigt)[0]
    Phi00 = Phi[(0,0,0)][ix]
    epsI += Phi00 * T(0,0,0,theta_obs,chi_obs,gamma_obs) * J00

    #Phi01 = Phi_generalized(np.array([x]), K=0, Kp=1, Q=0, vH=vH, a=a_voigt)[0]
    Phi01 = Phi[(0,1,0)][ix]
    epsV += Phi01 * T(3,1,0, theta_obs, chi_obs, gamma_obs) * J00
    epsV01 = Phi01 * T(3,1,0, theta_obs, chi_obs, gamma_obs) * J00
    V01_profile[ix] = np.real(epsV01)

    #Phi02 = Phi_generalized(np.array([x]), K=0, Kp=2, Q=0, vH=vH, a=a_voigt)[0]
    Phi02 = Phi[(0,2,0)][ix]
    epsI += Phi02 * T(0,2,0,theta_obs,chi_obs,gamma_obs) * J00
    epsQ += Phi02 * T(1,2,0,theta_obs,chi_obs,gamma_obs) * J00
    epsU += Phi02 * T(2,2,0,theta_obs,chi_obs,gamma_obs) * J00

    V21 = 0+0j
    epsV21 = 0j

    if RUN_DIAG and diag_is_sample_x(x):
        # Checklist #1: gate test for Q=0 channel in V
        gate_q0 = Phi01 * T(3,1,0,theta_obs,chi_obs,gamma_obs) * J00
        print("\n[DIAG #1] x =", x, "Q0 gate term =", gate_q0,
              "abs =", np.abs(gate_q0))

    c_plus = None
    c_minus = None

    # K=2 blocks
    for Q in [-2,-1,0,1,2]:
        phase = (-1)**Q
        rhoQ = np.conj(rho2[idx(-Q)])
        #phase = 1.0
        #rhoQ = rho2[idx(Q)]
        if Q == 0 and abs(x + 1) < 1e-10:
            old_debug_print("Dictionary value =", Phi[(2,1,0)][ix])
        Phi21 = Phi[(2,1,Q)][ix]
        Phi22 = Phi[(2,2,Q)][ix]
        Phi20 = Phi[(2,0,Q)][ix]

        #Phi20 = Phi_generalized(np.array([x]), K=2, Kp=0, Q=Q, vH=vH, a=a_voigt)[0]
        epsI += phase * Phi20 * T(0,0,0,theta_obs,chi_obs,gamma_obs) * rhoQ

        #Phi21 = Phi_generalized(np.array([x]), K=2, Kp=1, Q=Q, vH=vH, a=a_voigt)[0]
    
        # ---------------------------------------
        # DEBUG ONLY AT ONE FREQUENCY
        # ---------------------------------------
        if x == -2 or x == -1 or x == 0 or x == 1 or x == 2:
            old_debug_print(f"\nFrequency x={x}")
            term = (
                phase
                * Phi21
                * T(3,1,Q,theta_obs,chi_obs,gamma_obs)
                * rhoQ
            )
            old_debug_print(f"\nQ = {Q}")
            old_debug_print(f"RePhi21 = {np.real(Phi21)}")
            old_debug_print(f"ImPhi21 = {np.imag(Phi21)}")
            old_debug_print(f"ReT31    = {np.real(T(3,1,Q,theta_obs,chi_obs,gamma_obs))}")
            old_debug_print(f"ImT31    = {np.imag(T(3,1,Q,theta_obs,chi_obs,gamma_obs))}")
            old_debug_print(f"RerhoQ   = {np.real(rhoQ)}")
            old_debug_print(f"ImrhoQ   = {np.imag(rhoQ)}")
            old_debug_print(f"term   = {term}")
            old_debug_print(f"term.real   = {term.real}")
            old_debug_print(f"term.imag   = {term.imag}")
        epsI += phase * Phi21 * T(0,1,Q,theta_obs,chi_obs,gamma_obs) * rhoQ
        epsQ += phase * Phi21 * T(1,1,Q,theta_obs,chi_obs,gamma_obs) * rhoQ
        epsU += phase * Phi21 * T(2,1,Q,theta_obs,chi_obs,gamma_obs) * rhoQ
        epsV += phase * Phi21 * T(3,1,Q,theta_obs,chi_obs,gamma_obs) * rhoQ
        epsV21 += phase * Phi21 * T(3,1,Q,theta_obs,chi_obs,gamma_obs) * rhoQ

        if RUN_DIAG and diag_is_sample_x(x) and Q in [-1, 1]:
            pair_term = (
                phase
                * Phi21
                * T(3,1,Q,theta_obs,chi_obs,gamma_obs)
                * rhoQ
            )
            if Q == 1:
                c_plus = pair_term
            else:
                c_minus = pair_term

        if abs(x - 0.5) < 1e-10:
            V21 += phase * Phi21 * T(3,1,Q,theta_obs,chi_obs,gamma_obs) * rhoQ
            old_debug_print("V21 contribution for Q =", Q, "is", V21)

        #Phi22 = Phi_generalized(np.array([x]), K=2, Kp=2, Q=Q, vH=vH, a=a_voigt)[0]
        epsI += phase * Phi22 * T(0,2,Q,theta_obs,chi_obs,gamma_obs) * rhoQ
        epsQ += phase * Phi22 * T(1,2,Q,theta_obs,chi_obs,gamma_obs) * rhoQ
        epsU += phase * Phi22 * T(2,2,Q,theta_obs,chi_obs,gamma_obs) * rhoQ
        if Q == 1:
            V21_profile[ix] = (
                phase
                * Phi21
                * T(3,1,Q,theta_obs,chi_obs,gamma_obs)
                * rhoQ
            ).real
        if Q == 2:
            V21_2_profile[ix] = (
                phase
                * Phi21
                * T(3,1,Q,theta_obs,chi_obs,gamma_obs)
                * rhoQ
            ).real
        if Q == 0:
            V21_0_profile[ix] = (
                phase
                * Phi21
                * T(3,1,Q,theta_obs,chi_obs,gamma_obs)
                * rhoQ
            ).real
        if Q == -1:
            V21_m1_profile[ix] = (
                phase
                * Phi21
                * T(3,1,Q,theta_obs,chi_obs,gamma_obs)
                * rhoQ
            ).real
        if Q == -2:
            V21_m2_profile[ix] = (
                phase
                * Phi21
                * T(3,1,Q,theta_obs,chi_obs,gamma_obs)
                * rhoQ
            ).real

    if RUN_DIAG and diag_is_sample_x(x):
        # Checklist #2: pair-cancellation residual for Q=+/-1
        if c_plus is not None and c_minus is not None:
            pair_resid_plus = c_minus + np.conj(c_plus)
            pair_resid_minus = c_minus - np.conj(c_plus)
            pair_scale = max(np.abs(c_minus), np.abs(c_plus), 1e-300)
            print(
                "[DIAG #2] x =", x,
                "|c_- + c_+*| =", np.abs(pair_resid_plus),
                "rel_plus =", np.abs(pair_resid_plus)/pair_scale,
                "|c_- - c_+*| =", np.abs(pair_resid_minus),
                "rel_minus =", np.abs(pair_resid_minus)/pair_scale
            )

        # Checklist #5: micro-perturbation sensitivity
        v_ref = diag_v21_variant_at_ix(ix, rho2, use_conj_minusQ=True, flip_phase=False)
        v_no_conj = diag_v21_variant_at_ix(ix, rho2, use_conj_minusQ=False, flip_phase=False)
        v_flip_phase = diag_v21_variant_at_ix(ix, rho2, use_conj_minusQ=True, flip_phase=True)
        sref = np.abs(v_ref) + 1e-300

        print(
            "[DIAG #5] x =", x,
            "|v_ref| =", np.abs(v_ref),
            "phase(v_ref)[deg] =", complex_phase_deg(v_ref),
            "|v_no_conj|/|v_ref| =", np.abs(v_no_conj)/sref,
            "phase(v_no_conj)[deg] =", complex_phase_deg(v_no_conj),
            "|v_no_conj-v_ref|/|v_ref| =", np.abs(v_no_conj - v_ref)/sref,
            "|v_flip_phase|/|v_ref| =", np.abs(v_flip_phase)/sref,
            "phase(v_flip_phase)[deg] =", complex_phase_deg(v_flip_phase),
            "|v_flip_phase-v_ref|/|v_ref| =", np.abs(v_flip_phase - v_ref)/sref
        )

    #epsV = epsV01 + epsV21
    #print("x =", x)
    #print("epsV01 =", epsV01)
    #print("epsV21 =", epsV21)
    #print("epsV   =", epsV)

    i_val = np.real(epsI) * np.sqrt(np.pi)
    q_val = np.real(epsQ) * np.sqrt(np.pi)
    u_val = np.real(epsU) * np.sqrt(np.pi)
    v_val = np.real(epsV) * np.sqrt(np.pi)

    if np.abs(qu_back_rotation) > 0.0:
        q_val, u_val = _rotate_qu(q_val, u_val, qu_back_rotation)

    I_prof[ix] = i_val
    Q_prof[ix] = q_val
    U_prof[ix] = u_val
    V_prof[ix] = v_val

if RUN_DIAG:
    # Checklist #4: odd-parity residual of V(x)
    even_residual = np.max(np.abs(V_prof + V_prof[::-1]))
    odd_signal = np.max(np.abs(V_prof - V_prof[::-1])) + 1e-300
    print("\n[DIAG #4] max even residual =", even_residual)
    print("[DIAG #4] even/odd ratio =", even_residual / odd_signal)

    # Decomposition plot for vertical-field troubleshooting.
    v01 = np.sqrt(np.pi) * V01_profile
    v21m1 = np.sqrt(np.pi) * V21_m1_profile
    v210 = np.sqrt(np.pi) * V21_0_profile
    v21p1 = np.sqrt(np.pi) * V21_profile
    v21sum = v21m1 + v210 + v21p1

    fig_diag, ax_diag = plt.subplots(figsize=(9, 5))
    ax_diag.plot(xgrid, V_prof, color="k", linewidth=2.0, label="V total")
    ax_diag.plot(xgrid, v01, color="tab:gray", linestyle="--", label="K=0->K'=1, Q=0")
    ax_diag.plot(xgrid, v21m1, color="tab:blue", label="K=2->K'=1, Q=-1")
    ax_diag.plot(xgrid, v210, color="tab:green", label="K=2->K'=1, Q=0")
    ax_diag.plot(xgrid, v21p1, color="tab:orange", label="K=2->K'=1, Q=+1")
    ax_diag.plot(xgrid, v21sum, color="tab:red", linestyle=":", linewidth=2.0,
                 label="K=2->K'=1 sum")
    ax_diag.set_xlabel("Reduced frequency x")
    ax_diag.set_ylabel("V contribution")
    ax_diag.set_title("Stokes V decomposition (diagnostic)")
    ax_diag.grid(alpha=0.25)
    ax_diag.legend(fontsize=8, ncol=2)
    fig_diag.tight_layout()
    fig_diag.savefig("V_decomposition_diag.png", dpi=250)
    plt.close(fig_diag)

fig, ax = plt.subplots(2,2,figsize=(12,10))
fig.suptitle("Stokes profiles for Hu = {}, theta_B = {}, chi_B = {}, theta_obs = {}, chi_obs = {}".format(Hu, np.degrees(theta_B), chi_B, np.degrees(theta_obs), np.degrees(chi_obs)))
ax[0,0].plot(xgrid,I_prof)
ax[0,0].set_title("I")

ax[0,1].plot(xgrid,Q_prof)
ax[0,1].set_title("Q")

ax[1,0].plot(xgrid,U_prof)
ax[1,0].set_title("U")

ax[1,1].plot(xgrid,V_prof)
#ax[1,1].set_ylim(-0.00000002, 0.00000002)
ax[1,1].set_title("V")

plt.tight_layout()
plt.savefig("VStokes_try_Hu{}_gamma{}_thetaB{}_vH{}.png".format(Hu, np.degrees(gamma_obs), np.degrees(theta_B), vH), dpi = 300)
plt.close()

plot_fractional_polarization(
    xgrid,
    I_prof,
    Q_prof,
    U_prof,
    V_prof,
    "Fractional polarization (Fig. 13.7-like, generalized branch)",
    "Fractional_polarization_fig13_7_like_generalized.png"
)


Phi_test, components_test = Phi_generalized(
    xgrid,
    K=2,
    Kp=1,
    Q=0,
    vH=vH,
    a=a_voigt,
    return_pairs=True,
)

for Mu in (-1,0,1):
    for Mup in (-1,0,1):

        old_debug_print(
            "Mu", Mu,
            "Mup", Mup,
            "Abs max", np.max(np.abs(np.imag(components_test[Mu+1,Mup+1])))
        )

        plt.plot(
            xgrid,
            np.imag(components_test[Mu+1,Mup+1]),
            label=f"{Mu},{Mup}"
        )
plt.legend()
plt.savefig("VStokes_abs_Mu_Mup.png", dpi = 300)
plt.close()

Jarr = Jrad_to_array(Jrad_0)
J00 = Jrad_0[(0,0)]
Hfull = apply_hanle(Jarr, Hu, theta_B, 0.0)

np.set_printoptions(precision=6, suppress=True)

old_debug_print("Hfull for chi_B = 0, =")
old_debug_print(Hfull)

Hfull_pi2 = apply_hanle(Jarr, Hu, theta_B, -np.pi/2)
old_debug_print("Hfull for chi_B = -np.pi/2, =")
old_debug_print(Hfull_pi2)


D = wigner_D2(0.0, np.pi/2, 0.0)

np.set_printoptions(precision=6, suppress=True)
old_debug_print("D = ", D)

old_debug_print("Re(D) = ", np.real(D))
old_debug_print("Im(D) = ", np.imag(D))

old_debug_print(hanle_operator_alt(Hu, theta_B, chi_B)[:,2])


old_debug_print("----------------------------------------")
old_debug_print("theta_B = 90 deg, chi_B = 0.0 deg")
rho = apply_hanle(Jarr, Hu, np.pi/2, 0.0)
for Q in [-1,0,1]:
    rhoQ = rho[idx(Q)]
    contrib = (
        (-1.0)**Q
        * np.conj(rho[idx(-Q)])
        #rhoQ
        * Phi[(2,1,Q)]
        * T(3, 1, Q, theta_obs, chi_obs, gamma_obs)
    )

    old_debug_print("Q = ", Q, 
                    np.max(np.abs(contrib)),
                    np.max(np.abs(contrib.imag)))

old_debug_print("theta_B = 45 deg, chi_B = -90 deg")
rho = apply_hanle(Jarr, Hu, np.pi/4, -np.pi/2)
for Q in [-1,0,1]:
    rhoQ = rho[idx(Q)]
    contrib = (
        (-1.0)**Q
        * np.conj(rho[idx(-Q)])
        #rhoQ
        * Phi[(2,1,Q)]
        * T(3, 1, Q, theta_obs, chi_obs, gamma_obs)
    )

    old_debug_print("Q = ", Q,
                    np.max(np.abs(contrib)),
                    np.max(np.abs(contrib.imag)))

old_debug_print(T(3, 1, -1, theta_obs, chi_obs, gamma_obs))
old_debug_print(T(3, 1, 1, theta_obs, chi_obs, gamma_obs))
old_debug_print(T(3, 1, 0, theta_obs, chi_obs, gamma_obs))

V_terms = {}
V10   = np.zeros_like(xgrid)
V21m1 = np.zeros_like(xgrid)
V210  = np.zeros_like(xgrid)
V21p1 = np.zeros_like(xgrid)
for ix, x in enumerate(xgrid):
    J00 = J00_base
    rho2 = rho2_base

    epsI = 0.0 + 0j
    epsQ = 0.0 + 0j
    epsU = 0.0 + 0j
    epsV = 0.0 + 0j

    # ------------------------------------
    # K = 0 contributions
    # ------------------------------------

    Phi00 = Phi_appendix(
        np.array([x]), 0,0,0,vH,a_voigt
    )[0]

    epsI += (
        Phi00
        * T(0,0,0,theta_obs,chi_obs,gamma_obs)
        * J00
    )

    Phi01 = Phi_appendix(
        np.array([x]),0,1,0,vH,a_voigt
    )[0]

    term10 = (
        Phi01
        * T(3,1,0,theta_obs,chi_obs,gamma_obs)
        * J00
    )

    key = (0, 1, 0)

    if key not in V_terms:
        V_terms[key] = np.zeros_like(xgrid)

    V_terms[key][ix] = np.real(term10)

    Phi02 = Phi_appendix(
        np.array([x]),0,2,0,vH,a_voigt
    )[0]

    epsI += (
        Phi02
        * T(0,2,0,theta_obs,chi_obs,gamma_obs)
        * J00
    )

    epsQ += (
        Phi02
        * T(1,2,0,theta_obs,chi_obs,gamma_obs)
        * J00
    )

    epsU += (
        Phi02
        * T(2,2,0,theta_obs,chi_obs,gamma_obs)
        * J00
    )

    # ------------------------------------
    # K = 2 contributions
    # ------------------------------------
    epsV21 = 0.0 + 0j
    for Q in [-2,-1,0,1,2]:

        phase = (-1)**Q
        rhoQ  = np.conj(rho2[idx(-Q)])

        # -------------------------
        # K=2 -> K'=0
        # -------------------------

        Phi20 = Phi_appendix(
            np.array([x]),2,0,Q,vH,a_voigt
        )[0]

        epsI += (
            phase
            * Phi20
            * T(0,0,0,theta_obs,chi_obs,gamma_obs)
            * rhoQ
        )

        # -------------------------
        # K=2 -> K'=1
        # -------------------------

        Phi21 = Phi_appendix(
            np.array([x]),2,1,Q,vH,a_voigt
        )[0]

        term21I = (
            phase
            * Phi21
            * T(0,1,Q,theta_obs,chi_obs,gamma_obs)
            * rhoQ
        )

        term21Q = (
            phase
            * Phi21
            * T(1,1,Q,theta_obs,chi_obs,gamma_obs)
            * rhoQ
        )

        term21U = (
            phase
            * Phi21
            * T(2,1,Q,theta_obs,chi_obs,gamma_obs)
            * rhoQ
        )

        term21V = (
            phase
            * Phi21
            * T(3,1,Q,theta_obs,chi_obs,gamma_obs)
            * rhoQ
        )
        if abs(term21V) > 0:
            old_debug_print(
                "Q", Q,
                "term21V", term21V,
                np.max(np.abs(term21V)),
                "term21V real max", np.max(np.abs(np.real(term21V))),
                np.max(np.abs(np.real(term21V))),
                "term21V imag max",
                np.max(np.abs(np.imag(term21V)))
            )
        epsI += term21I
        epsQ += term21Q
        epsU += term21U
        epsV += term21V
        epsV21 += term21V
        # Store each individual V contribution

        if Q == -1:
            V21m1[ix] = np.real(term21V)

        elif Q == 0:
            V210[ix] = np.real(term21V)

        elif Q == 1:
            V21p1[ix] = np.real(term21V)

        key = (2, 1, Q)

        if key not in V_terms:
            V_terms[key] = np.zeros_like(xgrid)

        V_terms[key][ix] = np.real(term21V)

        # -------------------------
        # K=2 -> K'=2
        # -------------------------

        Phi22 = Phi_appendix(
            np.array([x]),2,2,Q,vH,a_voigt
        )[0]

        epsI += (
            phase
            * Phi22
            * T(0,2,Q,theta_obs,chi_obs,gamma_obs)
            * rhoQ
        )

        epsQ += (
            phase
            * Phi22
            * T(1,2,Q,theta_obs,chi_obs,gamma_obs)
            * rhoQ
        )

        epsU += (
            phase
            * Phi22
            * T(2,2,Q,theta_obs,chi_obs,gamma_obs)
            * rhoQ
        )

    # ------------------------------------
    # Store Stokes profiles
    # ------------------------------------
    check = V21m1[ix] + V210[ix] + V21p1[ix]

    old_debug_print(
        f"x = {x:6.2f}",
        f"stored = {check: .6e}",
        f"epsV21 = {np.real(epsV21): .6e}",
        f"difference = {check - np.real(epsV21): .3e}"
    )

    i_val = np.real(epsI) * np.sqrt(np.pi)
    q_val = np.real(epsQ) * np.sqrt(np.pi)
    u_val = np.real(epsU) * np.sqrt(np.pi)
    v_val = np.real(epsV) * np.sqrt(np.pi)

    if np.abs(qu_back_rotation) > 0.0:
        q_val, u_val = _rotate_qu(q_val, u_val, qu_back_rotation)

    I_prof[ix] = i_val
    Q_prof[ix] = q_val
    U_prof[ix] = u_val
    V_prof[ix] = v_val
fig, ax = plt.subplots(2,2,figsize=(12,10))
fig.suptitle("Stokes profiles for Hu = {}, theta_B = {}, chi_B = {}, theta_obs = {}, chi_obs = {}".format(Hu, np.degrees(theta_B), chi_B, np.degrees(theta_obs), np.degrees(chi_obs)))
ax[0,0].plot(xgrid,I_prof)
ax[0,0].set_title("I")

ax[0,1].plot(xgrid,Q_prof)
ax[0,1].set_title("Q")

ax[1,0].plot(xgrid,U_prof)
ax[1,0].set_title("U")

ax[1,1].plot(xgrid,V_prof)
ax[1,1].set_title("V")

plt.tight_layout()
plt.savefig("Stokes_B_Hu{}_gamma{}_thetaB{}_chiB{}_thetaobs{}_chiobs{}.png".format(Hu, np.degrees(gamma_obs), np.degrees(theta_B), np.degrees(chi_B), np.degrees(theta_obs), np.degrees(chi_obs)), dpi = 300)
plt.close()

plot_fractional_polarization(
    xgrid,
    I_prof,
    Q_prof,
    U_prof,
    V_prof,
    "Fractional polarization (Fig. 13.7-like, appendix branch)",
    "Fractional_polarization_fig13_7_like_appendix.png"
)

old_debug_print(np.max(np.abs(np.imag(Phi[(2,1,-1)]))))
old_debug_print(np.max(np.abs(np.real(Phi[(2,1,-1)]))))

Phi21 = Phi_generalized(
    xgrid,
    K=2,
    Kp=1,
    Q=1,
    vH=vH,
    a=a_voigt
)
phi_plus  = np.imag(phi_transition_complex(xgrid, 1, 0, vH, a_voigt))
phi_minus = np.imag(phi_transition_complex(xgrid,-1, 0, vH, a_voigt))
plt.figure()
#plt.plot(xgrid, phi_plus, label="phi_plus")
#plt.plot(xgrid, phi_minus, label="phi_minus")
plt.plot(xgrid, phi_plus-phi_minus, label="phi_plus-phi_minus")
plt.plot(xgrid, np.imag(Phi21), label="Phi_211")
plt.legend()
plt.savefig("Imag_diff.png")
plt.close()


Phi_gen_211 = Phi_generalized(
    xgrid,
    K=2,
    Kp=1,
    Q=1,
    vH=vH,
    a=a_voigt
)

Phi_gen_21m1 = Phi_generalized(
    xgrid,
    K=2,
    Kp=1,
    Q=-1,
    vH=vH,
    a=a_voigt
)

Phi_appendix_211 = Phi_appendix(xgrid, K = 2, Kp = 1, Q = 1, vH = vH, a = a_voigt)
Phi_appendix_21m1 = Phi_appendix(xgrid, K = 2, Kp = 1, Q = -1, vH = vH, a = a_voigt)

plt.figure(figsize=(8,6))
plt.plot(xgrid, np.real(Phi_appendix_211), color = "blue", linestyle = "--", label = r"$\Phi_1^{21'}(app)$")
plt.plot(xgrid, np.real(Phi_appendix_21m1), color = "navy", linestyle = "--", label = r"$\Phi_{-1}^{21'}(app)$")
plt.plot(xgrid, np.real(Phi_gen_211), color = "yellow", linestyle = ":", label = r"$\Phi_1^{21'}(gen)$")
plt.plot(xgrid, np.real(Phi_gen_21m1), color = "orange", linestyle = ":", label = r"$\Phi_{-1}^{21'}(gen)$")
plt.ylabel(r"$\mathrm{Re}\Phi_{\pm1}^{21'}$")
plt.xlabel("Reduced frequency x")
plt.legend()
plt.savefig("Gen_vs_appendix_Re.png")
plt.close()

plt.figure(figsize=(8,6))
plt.plot(xgrid, np.imag(Phi_appendix_211), color = "blue", linestyle = "--", label = r"$\Phi_1^{21'}(app)$")
plt.plot(xgrid, np.imag(Phi_appendix_21m1), color = "navy", linestyle = "--", label = r"$\Phi_{-1}^{21'}(app)$")
plt.plot(xgrid, np.imag(Phi_gen_211), color = "yellow", linestyle = ":", label = r"$\Phi_1^{21'}(gen)$")
plt.plot(xgrid, np.imag(Phi_gen_21m1), color = "orange", linestyle = ":", label = r"$\Phi_{-1}^{21'}(gen)$")
plt.ylabel(r"$\mathrm{Im}\Phi_{\pm1}^{21'}$")
plt.xlabel("Reduced frequency x")
plt.legend()
plt.savefig("Gen_vs_appendix_Im.png")
plt.close()

old_debug_print("Abs max, app im Phi_1^21'",np.abs(np.max(np.imag(Phi_appendix_211))))
old_debug_print("Abs max, app im Phi_-1^21'",np.abs(np.max(np.imag(Phi_appendix_21m1))))
old_debug_print("Abs max, gen im Phi_1^21'", np.abs(np.max(np.imag(Phi_gen_211))))
old_debug_print("Abs max, gen im Phi_-1^21'", np.abs(np.max(np.imag(Phi_gen_21m1))))

old_debug_print("Abs max, app re Phi_1^21'",np.abs(np.max(np.real(Phi_appendix_211))))
old_debug_print("Abs max, app re Phi_-1^21'",np.abs(np.max(np.real(Phi_appendix_21m1))))
old_debug_print("Abs max, gen re Phi_1^21'", np.abs(np.max(np.real(Phi_gen_211))))
old_debug_print("Abs max, gen re Phi_-1^21'", np.abs(np.max(np.real(Phi_gen_21m1))))

# Pogledati jednacine (3.38), (5.37), (5.45), (5.52), (6.59a), (9.6), (9.19)
# Gamma = A_ul/4*pi
# Raspisati sve T i Phi postupno za svaku kombinaciju K, K' i Q
d = wigner_d2(theta_B)
old_debug_print(np.round(np.real(d @ d.T),12))
old_debug_print(np.max(np.abs(d @ d.T - np.eye(5))))

rho = apply_hanle(Jarr, Hu, theta_B, 0.0)
old_debug_print(rho)

old_debug_print("rho[-2] ?", rho[0], " expected ", np.conj(rho[4]))
old_debug_print("rho[-1] ?", rho[1], " expected ", -np.conj(rho[3]))
old_debug_print("rho[0]  ?", rho[2])

for K in [0,1,2]:
    for Kp in [0,1,2]:
        for Q in range(-2,3):

            if abs(Q)>K or abs(Q)>Kp:
                continue

            P1 = Phi_generalized(
                x,
                K,Kp,Q,
                vH,a
            )

            P2 = Phi_appendix(
                x,
                K,Kp,Q,
                vH,a
            )

            err = np.max(np.abs(P1-P2))

            old_debug_print(
                K,Kp,Q,
                err
            )

# Hanle diagram for delta = 30 degrees
delta_ttt = np.radians(30.0)
Hu_ttt = 0.0
print("T for theta = np.pi/2:", T(1,2,0, np.pi/2, 0, 0))
print("T for theta = np.pi/2 - delta_ttt:", T(1,2,0, np.pi/2 - delta_ttt, 0, 0))

from Response_fun import *
profile_kind = "generalized"
phi_der = build_phi_table(xgrid, profile_kind=profile_kind, vH=vH, a_voigt=a_voigt)
B_array = np.linspace(0.0, 100.0, 100)

Jarr_base = Jrad_to_array(Jrad_0)
J00_base = Jrad_0[(0,0)]
qu_back_rotation = 0.0


J_rad0 = radiation_tensor(0.073)

hu = hu_default
theta_B = np.pi/2
chi_B = 0.0
theta_obs = np.pi/2
chi_obs = 0.0
gamma_obs = np.pi/2

state = prepare_magnetic_branch_state(
        J_rad0,
        hu,
        theta_B,
        chi_B,
        theta_obs,
        chi_obs,
        gamma_obs,
        USE_Q_U_REFERENCE_MODE,
    )

I_response, Q_response, U_response, V_response = response_function_as_derivative_B(xgrid, phi_der, state, B_array)
print("I_response shape:", I_response.shape)
print("Q_response shape:", Q_response.shape)
print("U_response shape:", U_response.shape)
print("V_response shape:", V_response.shape)

# Plot 2x2 with I, Q, U, V response functions
fig, ax = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle("Response functions for Hu = {}, theta_B = {}, chi_B = {}, theta_obs = {}, chi_obs = {}".format(hu, np.degrees(theta_B), chi_B, np.degrees(theta_obs), np.degrees(chi_obs)))
ax[0, 0].plot(B_array, I_response[:, 50])
ax[0, 0].set_title("I response")
ax[0, 1].plot(B_array, Q_response[:, 50])
ax[0, 1].set_title("Q response") 
ax[1, 0].plot(B_array, U_response[:, 50])
ax[1, 0].set_title("U response")
ax[1, 1].plot(B_array, V_response[:, 50])
ax[1, 1].set_title("V response")
plt.tight_layout()
plt.savefig("Response_functions_Hu{}_thetaB{}_chiB{}_thetaobs{}_chiobs{}.png".format(hu, np.degrees(theta_B), chi_B, np.degrees(theta_obs), np.degrees(chi_obs)), dpi=300)
plt.close()   

# response function for one specific B value
B_specific = 50.0
I_response_spec, Q_response_spec, U_response_spec, V_response_spec = response_function_B(xgrid, phi_der, state, B_specific)
fig, ax = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle("Response functions for B = {}, Hu = {}, theta_B = {}, chi_B = {}, theta_obs = {}, chi_obs = {}".format(B_specific, hu, np.degrees(theta_B), chi_B, np.degrees(theta_obs), np.degrees(chi_obs)))
ax[0, 0].plot(xgrid, I_response_spec)
ax[0, 0].set_title("I response")
ax[0, 1].plot(xgrid, Q_response_spec)
ax[0, 1].set_title("Q response")
ax[1, 0].plot(xgrid, U_response_spec)   
ax[1, 0].set_title("U response")
ax[1, 1].plot(xgrid, V_response_spec)
ax[1, 1].set_title("V response")
plt.tight_layout()
plt.savefig("Response_functions_B{}_Hu{}_thetaB{}_chiB{}_thetaobs{}_chiobs{}.png".format(B_specific, hu, np.degrees(theta_B), chi_B, np.degrees(theta_obs), np.degrees(chi_obs)), dpi=300)
plt.close()
print(I_response_spec)
print(Q_response_spec)
print(U_response_spec)
print(V_response_spec)

# We want to calculate Stokes for one specific B_value and use delta_B = 0.1 Gauss
# to compute the response function as a finite difference

# B array for finite difference response function
Bs_arr = np.linspace(5.69, 60.0, 100)
# Hu for each B value
Hus_arr = np.zeros_like(Bs_arr)
for B in Bs_arr:
    Hus_arr[np.where(Bs_arr == B)] = hanle_parameter_exact(B, gJu = 1.0, Aul = A_ul)
delta_B = 5.0 # perturbation

hp_array = np.linspace(0.0, 1.0, 100)
delta_0 = 0.0
hr_array = np.zeros_like(hp_array)
for hp in hp_array:
    hR = (1+hp)/np.cos(delta_0) - 1
    hr_array[np.where(hp_array == hp)] = hR
'''
Jrad_arr = np.zeros_like(hr_array)
for hr in hr_array:
    Jrad = radiation_tensor(hr)
    Jrad_arr[np.where(hr_array == hr)] = Jrad

# 1. Prepare states for one specific B and for all Jarr 
states_arr_const_B = []
for j in range(len(Jrad_arr)):
    B = 5.69
    Hu = Hus_arr[0]
    Jrad = Jrad_arr[j]
    state = prepare_magnetic_branch_state(
        Jrad,
        Hu,
        theta_B,
        chi_B,
        theta_obs,
        chi_obs,
        gamma_obs,
        USE_Q_U_REFERENCE_MODE,
    )
    states_arr_const_B.append(state)

I_response_fd, Q_response_fd, U_response_fd, V_response_fd = B_finite_difference_response(xgrid, phi_der, states_arr_const_B, Bs_arr, delta_B)
'''

def B_finite_difference_response_local(
    xgrid,
    phi,
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
    scheme="central",
    normalize="I",   # None, "I", or "self"
):
    if delta_B <= 0.0:
        raise ValueError("delta_B must be > 0.")

    # Build baseline and perturbed states (same geometry + same J at one height)
    hu0 = hanle_parameter_exact(B0, gJu, Aul)
    #print("hu0 =", hu0)
    state0 = prepare_magnetic_branch_state(
        jrad, hu0, theta_B, chi_B, theta_obs, chi_obs, gamma_obs, q_u_reference_mode
    )
    #print("state0 =", state0["rho2"])
    I0, Q0, U0, V0 = compute_stokes_profiles(xgrid, phi, state0)
    #print("I0 =", I0)

    hu_p = hanle_parameter_exact(B0 + delta_B, gJu, Aul)
    #print("hu_p =", hu_p)
    state_p = prepare_magnetic_branch_state(
        jrad, hu_p, theta_B, chi_B, theta_obs, chi_obs, gamma_obs, q_u_reference_mode
    )
    #print("state_p =", state_p["rho2"])
    Ip, Qp, Up, Vp = compute_stokes_profiles(xgrid, phi, state_p)
    #print("Ip =", Ip)

    if scheme == "central":
        hu_m = hanle_parameter_exact(B0 - delta_B, gJu, Aul)
        #print("hu_m =", hu_m)
        state_m = prepare_magnetic_branch_state(
            jrad, hu_m, theta_B, chi_B, theta_obs, chi_obs, gamma_obs, q_u_reference_mode
        )
        #print("state_m =", state_m["rho2"])
        Im, Qm, Um, Vm = compute_stokes_profiles(xgrid, phi, state_m)
        #print("Im =", Im)

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

hp_array = np.linspace(0.0, 1.0, 100)
h_true = (1.0 + hp_array) / np.cos(delta_0) - 1.0

Vresp_map = np.zeros((len(h_true), len(xgrid)))
Iresp_map = np.zeros((len(h_true), len(xgrid)))
for ih, hR in enumerate(hp_array):
    jrad = radiation_tensor(hr_array[ih])   # dict
    dIdB_relI, _, _, dVdB_relI, _, _, _, _ = B_finite_difference_response_local(
        xgrid=xgrid,
        phi=phi_der,
        jrad=jrad,
        B0=60.0,
        delta_B=10.0,
        theta_B=theta_B,
        chi_B=chi_B,
        theta_obs=theta_obs,
        chi_obs=chi_obs,
        gamma_obs=gamma_obs,
        q_u_reference_mode=Q_U_REFERENCE_MODE,
        scheme="central",
        normalize="I",
    )
    Vresp_map[ih, :] = dVdB_relI
    Iresp_map[ih, :] = dIdB_relI


plt.figure(figsize=(7, 6))
plt.imshow(
    Iresp_map,
    origin="lower",
    aspect="auto",
    extent=[xgrid[0], xgrid[-1], h_true[0], h_true[-1]],
    cmap="RdYlBu_r",
    vmin=-0.001, vmax=0.001
)
plt.xlabel("x")
plt.ylabel("h (true height)")
plt.colorbar(label="(1/I) dV/dB")
plt.tight_layout()
plt.savefig("RF_B_V_map.png", dpi=300)
print(Iresp_map)


delta = np.linspace(0,np.pi/2,200)

A1 = T(0,2,0,delta,0,np.pi/2)
A2 = T(0,2,0,np.pi/2-delta,0,np.pi/2)
plt.figure(figsize=(7,6))
plt.plot(np.degrees(delta), A1, label=r"$T(0,2,0,\delta,0,\pi/2)$")
plt.plot(np.degrees(delta), A2, label=r"$T(0,2,0,\pi/2-\delta,0,\pi/2)$")
plt.xlabel(r"$\delta$ [deg]")
plt.ylabel(r"$T(0,2,0)$")
plt.title("T(0,2,0) for delta = 0 to 90 degrees")
plt.legend()
plt.tight_layout()
plt.savefig("T_0_2_0_vs_delta.png", dpi=300)

# delta = 90 degrees
delta_case_90 = np.radians(90.0)
J_delta_case_90 = radiation_tensor(delta_case_90)
