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

theta_B = np.pi/2 # np.pi/4
chi_B = 0.0 # -np.pi/2
theta_obs = np.pi/2
chi_obs = 0.0
gamma_obs = np.pi/2

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

V21_profile = np.zeros_like(xgrid)
V21_2_profile = np.zeros_like(xgrid)
V21_0_profile = np.zeros_like(xgrid)
V21_m2_profile = np.zeros_like(xgrid)
V21_m1_profile = np.zeros_like(xgrid)
for ix, x in enumerate(xgrid):
    Jarr = Jrad_to_array(Jrad_0)
    J00 = Jrad_0[(0,0)]
    rho2 = apply_hanle(Jarr, Hu, theta_B, chi_B)
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

    #Phi02 = Phi_generalized(np.array([x]), K=0, Kp=2, Q=0, vH=vH, a=a_voigt)[0]
    Phi02 = Phi[(0,2,0)][ix]
    epsI += Phi02 * T(0,2,0,theta_obs,chi_obs,gamma_obs) * J00
    epsQ += Phi02 * T(1,2,0,theta_obs,chi_obs,gamma_obs) * J00
    epsU += Phi02 * T(2,2,0,theta_obs,chi_obs,gamma_obs) * J00

    V21 = 0+0j
    epsV21 = 0j
    # K=2 blocks
    for Q in [-2,-1,0,1,2]:
        phase = (-1)**Q
        rhoQ = np.conj(rho2[idx(-Q)])
        #phase = 1
        #rhoQ = rho2[idx(Q)]
        if Q == 0 and abs(x + 1) < 1e-10:
            print("Dictionary value =", Phi[(2,1,0)][ix])
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
            print(f"\nFrequency x={x}")
            term = (
                phase
                * Phi21
                * T(3,1,Q,theta_obs,chi_obs,gamma_obs)
                * rhoQ
            )
            print(f"\nQ = {Q}")
            print(f"RePhi21 = {np.real(Phi21)}")
            print(f"ImPhi21 = {np.imag(Phi21)}")
            print(f"ReT31    = {np.real(T(3,1,Q,theta_obs,chi_obs,gamma_obs))}")
            print(f"ImT31    = {np.imag(T(3,1,Q,theta_obs,chi_obs,gamma_obs))}")
            print(f"RerhoQ   = {np.real(rhoQ)}")
            print(f"ImrhoQ   = {np.imag(rhoQ)}")
            print(f"term   = {term}")
            print(f"term.real   = {term.real}")
            print(f"term.imag   = {term.imag}")
        epsI += phase * Phi21 * T(0,1,Q,theta_obs,chi_obs,gamma_obs) * rhoQ
        epsQ += phase * Phi21 * T(1,1,Q,theta_obs,chi_obs,gamma_obs) * rhoQ
        epsU += phase * Phi21 * T(2,1,Q,theta_obs,chi_obs,gamma_obs) * rhoQ
        epsV += phase * Phi21 * T(3,1,Q,theta_obs,chi_obs,gamma_obs) * rhoQ
        epsV21 += phase * Phi21 * T(3,1,Q,theta_obs,chi_obs,gamma_obs) * rhoQ
        if abs(x - 0.5) < 1e-10:
            V21 += phase * Phi21 * T(3,1,Q,theta_obs,chi_obs,gamma_obs) * rhoQ
            print("V21 contribution for Q =", Q, "is", V21)

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

    #epsV = epsV01 + epsV21
    #print("x =", x)
    #print("epsV01 =", epsV01)
    #print("epsV21 =", epsV21)
    #print("epsV   =", epsV)

    I_prof[ix] = (np.real(epsI)*np.sqrt(np.pi))
    Q_prof[ix] = (np.real(epsQ)*np.sqrt(np.pi))
    U_prof[ix] = (np.real(epsU)*np.sqrt(np.pi))
    V_prof[ix] = (np.real(epsV)*np.sqrt(np.pi))

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

        print(
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

print("Hfull for chi_B = 0, =")
print(Hfull)

Hfull_pi2 = apply_hanle(Jarr, Hu, theta_B, -np.pi/2)
print("Hfull for chi_B = -np.pi/2, =")
print(Hfull_pi2)


D = wigner_D2(0.0, np.pi/2, 0.0)

np.set_printoptions(precision=6, suppress=True)
print("D = ", D)

print("Re(D) = ", np.real(D))
print("Im(D) = ", np.imag(D))

print(hanle_operator_alt(Hu, theta_B, chi_B)[:,2])


print("----------------------------------------")
print("theta_B = 90 deg, chi_B = 0.0 deg")
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

    print("Q = ", Q, 
          np.max(np.abs(contrib)),
          np.max(np.abs(contrib.imag)))

print("theta_B = 45 deg, chi_B = -90 deg")
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

    print("Q = ", Q,
          np.max(np.abs(contrib)),
          np.max(np.abs(contrib.imag)))

print(T(3, 1, -1, theta_obs, chi_obs, gamma_obs))
print(T(3, 1, 1, theta_obs, chi_obs, gamma_obs))
print(T(3, 1, 0, theta_obs, chi_obs, gamma_obs))

V_terms = {}
V10   = np.zeros_like(xgrid)
V21m1 = np.zeros_like(xgrid)
V210  = np.zeros_like(xgrid)
V21p1 = np.zeros_like(xgrid)
for ix, x in enumerate(xgrid):

    Jarr = Jrad_to_array(Jrad_0)
    J00  = Jrad_0[(0,0)]

    rho2 = apply_hanle(Jarr, Hu, theta_B, chi_B)

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
            print(
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

    print(
        f"x = {x:6.2f}",
        f"stored = {check: .6e}",
        f"epsV21 = {np.real(epsV21): .6e}",
        f"difference = {check - np.real(epsV21): .3e}"
    )

    I_prof[ix] = np.real(epsI) * np.sqrt(np.pi)
    Q_prof[ix] = np.real(epsQ) * np.sqrt(np.pi)
    U_prof[ix] = np.real(epsU) * np.sqrt(np.pi)
    V_prof[ix] = np.real(epsV) * np.sqrt(np.pi)
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

print(np.max(np.abs(np.imag(Phi[(2,1,-1)]))))
print(np.max(np.abs(np.real(Phi[(2,1,-1)]))))

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

print("Abs max, app im Phi_1^21'",np.abs(np.max(np.imag(Phi_appendix_211))))
print("Abs max, app im Phi_-1^21'",np.abs(np.max(np.imag(Phi_appendix_21m1))))
print("Abs max, gen im Phi_1^21'", np.abs(np.max(np.imag(Phi_gen_211))))
print("Abs max, gen im Phi_-1^21'", np.abs(np.max(np.imag(Phi_gen_21m1))))

print("Abs max, app re Phi_1^21'",np.abs(np.max(np.real(Phi_appendix_211))))
print("Abs max, app re Phi_-1^21'",np.abs(np.max(np.real(Phi_appendix_21m1))))
print("Abs max, gen re Phi_1^21'", np.abs(np.max(np.real(Phi_gen_211))))
print("Abs max, gen re Phi_-1^21'", np.abs(np.max(np.real(Phi_gen_21m1))))

# Pogledati jednacine (3.38), (5.37), (5.45), (5.52), (6.59a), (9.6), (9.19)
# Gamma = A_ul/4*pi
# Raspisati sve T i Phi postupno za svaku kombinaciju K, K' i Q