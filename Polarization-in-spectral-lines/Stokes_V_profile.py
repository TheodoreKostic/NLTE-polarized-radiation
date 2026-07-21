import numpy as np
import sys
import os
import matplotlib.pyplot as plt

#script_dir = os.path.abspath("/home/Code/NLTE-polarized-radiation")
script_dir = os.path.abspath("/home/teodor/Documents/Codes/NLTE-polarized-radiation")
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

theta_B = np.pi/2 # -
chi_B = 0.0 # np.pi/4
theta_obs = np.pi/2
chi_obs = 0.0
gamma_obs = np.pi/2

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

# ----------------------------------------------------------
# Plot all generalized profiles
# ----------------------------------------------------------

styles = {
    -2: ("tab:purple", ":"),
    -1: ("tab:red", "--"),
     0: ("tab:green", "-"),
     1: ("tab:orange", "-."),
     2: ("tab:blue", (0, (5, 1)))
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

        # unsplit profile
        ax.plot(
            xgrid,
            np.real(phi_complex(xgrid,a_voigt)),
            "k--",
            alpha=0.35,
            linewidth=1.0,
            label="Voigt" if Q==Qlist[0] else None
        )

        # Zeeman components
        for Mu in (-1,0,1):
            for Mup in (-1,0,1):

                profile = components[Mu+1, Mup+1]

                ax.plot(
                    xgrid,
                    np.real(profile),
                    linewidth=1.2,
                    label=rf"$M_u={Mu},\,M_u'={Mup}$"
                )

        # total generalized profile
        ax.plot(
            xgrid,
            np.real(Phi),
            "k",
            linewidth=3,
            label=r"$\Phi^{KK'}_Q$"
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

        for Mu in (-1,0,1):
            for Mup in (-1,0,1):

                profile = components[Mu+1, Mup+1]

                ax.plot(
                    xgrid,
                    np.imag(profile),
                    linewidth=1.2,
                    label=rf"$M_u={Mu},\,M_u'={Mup}$"
                )

        ax.plot(
            xgrid,
            np.imag(Phi),
            "k",
            linewidth=3,
            label=r"$\Phi^{KK'}_Q$"
        )

    ax.grid(alpha=0.3)

    if row == 0:
        ax.legend(
            loc="center left",
            bbox_to_anchor=(1.02,0.5),
            fontsize=8,
            frameon=True
        )

axes[-1,0].set_xlabel("Reduced frequency $x$")
axes[-1,1].set_xlabel("Reduced frequency $x$")

fig.suptitle("Generalized profiles", fontsize=16)

plt.savefig(
    "Generalized_profiles_debug.png",
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