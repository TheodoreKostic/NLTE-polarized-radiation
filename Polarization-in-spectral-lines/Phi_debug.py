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

def emissivity_breakdown(
        x,
        Hu,
        Jrad,
        theta_B,
        chi_B,
        theta_obs,
        chi_obs,
        vH,
        gamma_obs=np.pi/2):

    J00 = Jrad[(0,0)]

    Jvert = Jrad_to_array(Jrad)

    rho = apply_hanle(
        Jvert,
        Hu,
        theta_B,
        chi_B
    )

    blocks = {
        "00": {"I":0j, "Q":0j, "U":0j},
        "02": {"I":0j, "Q":0j, "U":0j},
        "20": {"I":0j, "Q":0j, "U":0j},
        "22": {"I":0j, "Q":0j, "U":0j},
    }

    # =====================================================
    # (0,0)
    # =====================================================

    Phi00 = Phi_generalized(
        np.array([x]),
        K=0,
        Kp=0,
        Q=0,
        vH=vH
    )[0]

    blocks["00"]["I"] += (
        Phi00
        * T(0,0,0,
            theta_obs,
            chi_obs,
            gamma_obs)
        * J00
    )

    # =====================================================
    # (0,2)
    # =====================================================

    Phi02 = Phi_generalized(
        np.array([x]),
        K=0,
        Kp=2,
        Q=0,
        vH=vH
    )[0]

    blocks["02"]["I"] += (
        Phi02
        * T(0,2,0,
            theta_obs,
            chi_obs,
            gamma_obs)
        * J00
    )

    blocks["02"]["Q"] += (
        Phi02
        * T(1,2,0,
            theta_obs,
            chi_obs,
            gamma_obs)
        * J00
    )

    blocks["02"]["U"] += (
        Phi02
        * T(2,2,0,
            theta_obs,
            chi_obs,
            gamma_obs)
        * J00
    )

    # =====================================================
    # K=2 terms
    # =====================================================

    for Q in [-2,-1,0,1,2]:

        phase = (-1)**Q

        rhoQ = rho[idx(-Q)]

        # -----------------------------
        # (2,0)
        # -----------------------------

        Phi20 = Phi_generalized(
            np.array([x]),
            K=2,
            Kp=0,
            Q=Q,
            vH=vH
        )[0]

        blocks["20"]["I"] += (
            phase
            * Phi20
            * T(0,0,0,
                theta_obs,
                chi_obs,
                gamma_obs)
            * rhoQ
        )

        # -----------------------------
        # (2,2)
        # -----------------------------

        Phi22 = Phi_generalized(
            np.array([x]),
            K=2,
            Kp=2,
            Q=Q,
            vH=vH
        )[0]

        blocks["22"]["I"] += (
            phase
            * Phi22
            * T(0,2,Q,
                theta_obs,
                chi_obs,
                gamma_obs)
            * rhoQ
        )

        blocks["22"]["Q"] += (
            phase
            * Phi22
            * T(1,2,Q,
                theta_obs,
                chi_obs,
                gamma_obs)
            * rhoQ
        )

        blocks["22"]["U"] += (
            phase
            * Phi22
            * T(2,2,Q,
                theta_obs,
                chi_obs,
                gamma_obs)
            * rhoQ
        )

    print("\n===================================")
    print("EMISSIVITY BREAKDOWN")
    print("x =", x)
    print("===================================")

    Itot = Qtot = Utot = 0j

    for name in ["00","02","20","22"]:

        I = blocks[name]["I"]
        Q = blocks[name]["Q"]
        U = blocks[name]["U"]

        Itot += I
        Qtot += Q
        Utot += U

        print(f"\nBlock ({name[0]},{name[1]})")
        print("I =", np.real(I))
        print("Q =", np.real(Q))
        print("U =", np.real(U))

    print("\nTOTAL")
    print("I =", np.real(Itot))
    print("Q =", np.real(Qtot))
    print("U =", np.real(Utot))

    print("Q/I =", np.real(Qtot)/np.real(Itot))
    print("U/I =", np.real(Utot)/np.real(Itot))

Jrad_0 = radiation_tensor(hR=0.073)
Hu = 1.0
vH = 0.002

xgrid = np.linspace(-5,5,401)

I_prof = np.zeros_like(xgrid)
Q_prof = np.zeros_like(xgrid)
U_prof = np.zeros_like(xgrid)
V_prof = np.zeros_like(xgrid)

theta_B = np.pi/2
chi_B = 0.0
theta_obs = np.pi/2
chi_obs = 0.0
gamma_obs = np.pi/2
'''
for ix, x in enumerate(xgrid):
    Jarr = Jrad_to_array(Jrad_0)
    rho00 = Jrad_0[(0,0)]
    rho20 = apply_hanle(
        Jarr,
        Hu,
        theta_B,
        chi_B
    )
    epsI = 0.0+0j
    epsQ = 0.0+0j
    epsU = 0.0+0j
    epsV = 0.0+0j
    for K in [0, 1, 2]:
        for Kp in [0, 1, 2]:
            for Q in [-2,-1,0,1,2]:

                rho = rho20[idx(-Q)]

                phi22 = Phi_generalized(
                    np.array([x]),
                    K=K,
                    Kp=Kp,
                    Q=Q,
                    vH=0.002
                )[0]

                epsI += (
                    phi22
                    * (-1.0)**Q
                    * T(0,Kp,Q,theta_obs,chi_obs,np.pi/2)
                    * rho
                )

                epsQ += (
                    phi22
                    * (-1.0)**Q
                    * T(1,Kp,Q,theta_obs,chi_obs,np.pi/2)
                    * rho
                )

                epsU += (
                    phi22
                    * (-1.0)**Q
                    * T(2,Kp,Q,theta_obs,chi_obs,np.pi/2)
                    * rho
                )
                
                epsV +=(
                    phi22
                    * (-1.0)**Q
                    * T(3,Kp,Q,theta_obs,chi_obs,np.pi/2)
                    * rho
                )
            
            print("K = {}, Kp = {}, Q = {}".format(K, Kp, Q))
            print("Eps I", np.real(epsI))
            print("Eps Q", np.real(epsQ))
            print("Eps U", np.real(epsU))
            print(np.shape(np.real(epsV)))
            print("Eps V", np.real(epsV))
            
            # all Ks
            if K == 2:
                I_prof[ix] = (np.real(epsI[0]))
                Q_prof[ix] = (np.real(epsQ[0]))
                U_prof[ix] = (np.real(epsU[0]))
            elif K == 1:
                V_prof[ix] = (np.real(epsV[0]))
            else:
                I_prof[ix] = (np.real(epsI))
                Q_prof[ix] = (np.real(epsQ))
                U_prof[ix] = (np.real(epsU))
                V_prof[ix] = (np.real(epsV))
            
        I_prof[ix] = (np.real(epsI))
        Q_prof[ix] = (np.real(epsQ))
        U_prof[ix] = (np.real(epsU))
        V_prof[ix] = np.real(epsV)
'''           
for ix, x in enumerate(xgrid):
    Jarr = Jrad_to_array(Jrad_0)
    J00 = Jrad_0[(0,0)]
    rho2 = apply_hanle(Jarr, Hu, theta_B, chi_B)
    epsI = 0.0+0j
    epsQ = 0.0+0j
    epsU = 0.0+0j
    epsV = 0.0+0j
    # K=0 blocks
    Phi00 = Phi_generalized(np.array([x]), K=0, Kp=0, Q=0, vH=vH)[0]
    epsI += Phi00 * T(0,0,0,theta_obs,chi_obs,gamma_obs) * J00

    Phi02 = Phi_generalized(np.array([x]), K=0, Kp=2, Q=0, vH=vH)[0]
    epsI += Phi02 * T(0,2,0,theta_obs,chi_obs,gamma_obs) * J00
    epsQ += Phi02 * T(1,2,0,theta_obs,chi_obs,gamma_obs) * J00
    epsU += Phi02 * T(2,2,0,theta_obs,chi_obs,gamma_obs) * J00

    # K=2 blocks
    for Q in [-2,-1,0,1,2]:
        phase = (-1)**Q
        rhoQ = rho2[idx(-Q)]

        Phi20 = Phi_generalized(np.array([x]), K=2, Kp=0, Q=Q, vH=vH)[0]
        epsI += phase * Phi20 * T(0,0,0,theta_obs,chi_obs,gamma_obs) * rhoQ

        Phi21 = Phi_generalized(np.array([x]), K=2, Kp=1, Q=Q, vH=vH)[0]
        epsI += phase * Phi21 * T(0,1,Q,theta_obs,chi_obs,gamma_obs) * rhoQ
        epsQ += phase * Phi21 * T(1,1,Q,theta_obs,chi_obs,gamma_obs) * rhoQ
        epsU += phase * Phi21 * T(2,1,Q,theta_obs,chi_obs,gamma_obs) * rhoQ
        epsV += phase * Phi21 * T(3,1,Q,theta_obs,chi_obs,gamma_obs) * rhoQ

        Phi22 = Phi_generalized(np.array([x]), K=2, Kp=2, Q=Q, vH=vH)[0]
        epsI += phase * Phi22 * T(0,2,Q,theta_obs,chi_obs,gamma_obs) * rhoQ
        epsQ += phase * Phi22 * T(1,2,Q,theta_obs,chi_obs,gamma_obs) * rhoQ
        epsU += phase * Phi22 * T(2,2,Q,theta_obs,chi_obs,gamma_obs) * rhoQ
            
        I_prof[ix] = (np.real(epsI))
        Q_prof[ix] = (np.real(epsQ))
        U_prof[ix] = (np.real(epsU))
        V_prof[ix] = np.real(epsV)

fig, ax = plt.subplots(2,2,figsize=(10,8))

ax[0,0].plot(xgrid,I_prof)
ax[0,0].set_title("I")

ax[0,1].plot(xgrid,Q_prof)
ax[0,1].set_title("Q")

ax[1,0].plot(xgrid,U_prof)
ax[1,0].set_title("U")

ax[1,1].plot(xgrid,V_prof)
ax[1,1].set_title("V")

plt.tight_layout()
plt.savefig("Stokes_try_gamma_90_theta_90.png", dpi = 300)

emissivity_breakdown(
    x=0.0,
    Hu=1.0,
    Jrad=Jrad_0,
    theta_B=np.pi/2,
    chi_B=0.0,
    theta_obs=np.pi/2,
    chi_obs=0.0,
    vH=0.002
)

Phi_Q0 = Phi_generalized(
    xgrid,
    K=2,
    Kp=1,
    Q=0,
    vH=1.0
)
Phi_Q1 = Phi_generalized(
    xgrid,
    K=2,
    Kp=1,
    Q=1,
    vH=1.0
)
Phi_Q2 = Phi_generalized(
    xgrid,
    K=2,
    Kp=1,
    Q=2,
    vH=1.0
)
plt.figure()
plt.plot(xgrid, Phi_Q0, label="Q=0")
plt.plot(xgrid, Phi_Q1, label="Q=1")
plt.plot(xgrid, Phi_Q2, label="Q=2")
plt.xlabel("x")
plt.ylabel("Phi")
plt.title("Phi vs x for different Q values")
plt.legend()
plt.savefig("Phi_debug_1.png", dpi = 300)

for K in [0,1,2]:
    for Kp in [0,1,2]:
        val = Phi_generalized(
            np.array([0.0]),
            K,
            Kp,
            0,
            vH
        )[0]

        print(K,Kp,val)

for Q in [-2,-1,0,1,2]:
    print(Q,
          Phi_generalized(
             np.array([0]),
             2,
             1,
             Q,
             vH
          )[0])
    
print("Last suspicious check:")
for K in [0,1,2]:
    for Kp in [0,1,2]:
        for Q in [-2,-1,0,1,2]:

            val = Phi_generalized(
                np.array([0.0]),
                K,
                Kp,
                Q,
                vH=1.0
            )[0]

            if abs(val) > 1e-10:
                print(K,Kp,Q,val)

# Small test
def tP(i, P):

    if i == 0:

        if P == 0:
            return 1/np.sqrt(2)

    elif i == 1:

        if P == -2:
            return -np.sqrt(3)/2

        if P == 2:
            return -np.sqrt(3)/2

    elif i == 2:

        if P == -2:
            return +1j*np.sqrt(3)/2

        if P == 2:
            return -1j*np.sqrt(3)/2

    return 0.0j

def T_book(i, Q,
           theta_obs,
           chi_obs,
           gamma_obs):

    Dobs = wigner_D2(
        chi_obs,
        theta_obs,
        gamma_obs
    )

    val = 0j

    for P in [-2,-1,0,1,2]:

        val += (
            tP(i,P)
            *
            Dobs[idx(P),idx(Q)]
        )

    return val

theta = np.pi/2; chi = 0.0
for gamma in [0.0, np.pi/2]:
    print("gamma =", gamma)
    for i,name in [(0,"I"),(1,"Q"),(2,"U")]:
        for Q in [-2,-1,0,1,2]:
            a = T(i, {0:0,1:2}[i] if i>0 else 0, Q, theta, chi, gamma) if False else T(i,2,Q,theta,chi,gamma)  # call T same way you use it
            b = T_book(i,Q,theta,chi,gamma)
            print(i,Q, np.round(a,8), np.round(b,8), "diff", np.round(a-b,8))


x = np.array([0.0])
for sign in [1,-1]:
    # temporarily edit phi_transition_complex to use (sign*shift - x) or run an alternative function that flips arg
    P = Phi_generalized(x, K=2, Kp=1, Q=0, vH=1.0)  # run before/after edit
    print("sign", sign, "Phi(K=2,K'=1,Q=0) =", P[0])

rho = apply_hanle(Jarr, Hu=1.0, theta_B=np.pi/2, chi_B=0.0)
for Q in [-2,-1,0,1,2]:
    lhs = rho[idx(Q)]
    rhs = (-1)**Q * np.conj(rho[idx(-Q)])
    print(Q, lhs, rhs, "diff", lhs-rhs)

# from Ch13_various_tests.py example
rho = np.zeros(5, dtype=complex)
rho[idx(-2)] = -0.30618621784789724
rho[idx(0)]  = 0.25
rho[idx(+2)] = -0.30618621784789724

theta_obs = np.pi/2; chi_obs = 0.0; gamma_obs = np.pi/2
epsQ = 0j; epsU = 0j
for Q in [-2,-1,0,1,2]:
    termQ = T(1,2,Q,theta_obs,chi_obs,gamma_obs) * rho[idx(Q)]
    termU = T(2,2,Q,theta_obs,chi_obs,gamma_obs) * rho[idx(Q)]
    print("Q",Q,"termQ",termQ,"termU",termU)
    epsQ += termQ; epsU += termU

print("epsQ",epsQ,"epsU",epsU,"Q/I",np.real(epsQ/1.0),"U/I",np.real(epsU/1.0))

def tP(i, P):
    if i == 0:
        if P == 0:
            return 1.0 / np.sqrt(2)
    elif i == 1:
        if P == -2:
            return -np.sqrt(3) / 2
        if P == 2:
            return -np.sqrt(3) / 2
    elif i == 2:
        if P == -2:
            return +1j * np.sqrt(3) / 2
        if P == 2:
            return -1j * np.sqrt(3) / 2
    return 0.0 + 0.0j


def T_book(i, Q, theta_obs, chi_obs, gamma_obs):
    Dobs = wigner_D2(chi_obs, theta_obs, gamma_obs)
    val = 0j
    for P in [-2, -1, 0, 1, 2]:
        val += tP(i, P) * Dobs[idx(P), idx(Q)]
    return val


def compare_T_vs_T_book(theta_obs, chi_obs):
    def fmt(z):
        return f"{z.real:+.6f}{z.imag:+.6f}j"

    print("\n=== T vs T_book comparison for K'=2 tensors ===")
    print(f"{'gamma':>6} {'i':>2} {'Q':>2} {'T':>24} {'T_book':>24} {'diff':>24}")
    for gamma in [0.0, np.pi / 2]:
        for i in [0, 1, 2]:
            for Q in [-2, -1, 0, 1, 2]:
                t_val = T(i, 2, Q, theta_obs, chi_obs, gamma)
                tb_val = T_book(i, Q, theta_obs, chi_obs, gamma)
                diff = t_val - tb_val
                print(
                    f"{gamma:6.2f} {i:2d} {Q:2d} "
                    f"{fmt(t_val):>24} {fmt(tb_val):>24} {fmt(diff):>24}"
                )


def compute_stokes_profiles(
        theta_obs,
        chi_obs,
        gamma_obs,
        use_T_book=False):

    Jarr = Jrad_to_array(Jrad_0)
    rho2 = apply_hanle(Jarr, Hu, theta_B, chi_B)

    I_prof = np.zeros_like(xgrid)
    Q_prof = np.zeros_like(xgrid)
    U_prof = np.zeros_like(xgrid)
    V_prof = np.zeros_like(xgrid)

    def T_choice(i, Kp, Q):
        if use_T_book and Kp == 2 and i in (0, 1, 2):
            return T_book(i, Q, theta_obs, chi_obs, gamma_obs)
        return T(i, Kp, Q, theta_obs, chi_obs, gamma_obs)

    for ix, x in enumerate(xgrid):
        J00 = Jrad_0[(0, 0)]
        epsI = 0.0 + 0j
        epsQ = 0.0 + 0j
        epsU = 0.0 + 0j
        epsV = 0.0 + 0j

        Phi00 = Phi_generalized(np.array([x]), K=0, Kp=0, Q=0, vH=vH)[0]
        epsI += Phi00 * T_choice(0, 0, 0) * J00

        Phi02 = Phi_generalized(np.array([x]), K=0, Kp=2, Q=0, vH=vH)[0]
        epsI += Phi02 * T_choice(0, 2, 0) * J00
        epsQ += Phi02 * T_choice(1, 2, 0) * J00
        epsU += Phi02 * T_choice(2, 2, 0) * J00

        for Q in [-2, -1, 0, 1, 2]:
            phase = (-1) ** Q
            rhoQ = rho2[idx(-Q)]

            Phi20 = Phi_generalized(np.array([x]), K=2, Kp=0, Q=Q, vH=vH)[0]
            epsI += phase * Phi20 * T_choice(0, 0, Q) * rhoQ

            Phi21 = Phi_generalized(np.array([x]), K=2, Kp=1, Q=Q, vH=vH)[0]
            epsI += phase * Phi21 * T_choice(0, 1, Q) * rhoQ
            epsQ += phase * Phi21 * T_choice(1, 1, Q) * rhoQ
            epsU += phase * Phi21 * T_choice(2, 1, Q) * rhoQ
            epsV += phase * Phi21 * T(3, 1, Q, theta_obs, chi_obs, gamma_obs) * rhoQ

            Phi22 = Phi_generalized(np.array([x]), K=2, Kp=2, Q=Q, vH=vH)[0]
            epsI += phase * Phi22 * T_choice(0, 2, Q) * rhoQ
            epsQ += phase * Phi22 * T_choice(1, 2, Q) * rhoQ
            epsU += phase * Phi22 * T_choice(2, 2, Q) * rhoQ

        I_prof[ix] = np.real(epsI)
        Q_prof[ix] = np.real(epsQ)
        U_prof[ix] = np.real(epsU)
        V_prof[ix] = np.real(epsV)

    return I_prof, Q_prof, U_prof, V_prof


compare_T_vs_T_book(theta_obs, chi_obs)

I_T, Q_T, U_T, V_T = compute_stokes_profiles(
    theta_obs, chi_obs, gamma_obs, use_T_book=False
)
I_B, Q_B, U_B, V_B = compute_stokes_profiles(
    theta_obs, chi_obs, gamma_obs, use_T_book=True
)

fig, ax = plt.subplots(2, 2, figsize=(10, 8))

for row, label, data_T, data_B in [
        (0, "I", I_T, I_B),
        (1, "Q", Q_T, Q_B),
        (2, "U", U_T, U_B),
        (3, "V", V_T, V_B),
]:
    r = row // 2
    c = row % 2
    ax[r, c].plot(xgrid, data_T, "r-", label="T")
    ax[r, c].plot(xgrid, data_B, "b--", label="T_book")
    ax[r, c].set_title(label)
    ax[r, c].legend()

plt.tight_layout()
plt.savefig("Stokes_compare_T_Tbook.png", dpi=300)
print("Saved: Stokes_compare_T_Tbook.png")