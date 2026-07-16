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

print("phi_complex(0, a=0):", phi_complex(0.0, a=0.0))
print("phi_complex(0, a=1):", phi_complex(0.0, a=1.0))

default_Delta_nu_D = 4 * 10**9 # s^-1
B = 5.69 # Gauss
x = np.array([0.0])
#vH = 0.002
vH = 1.3996e6 * B / default_Delta_nu_D


P0 = Phi_generalized(x, K=2, Kp=2, Q=0, vH=vH, a=0.0)[0]
P1 = Phi_generalized(x, K=2, Kp=2, Q=0, vH=vH, a=1.0)[0]

print("a=0:", P0)
print("a=1:", P1)
print("diff:", P1 - P0)


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
        vH=vH,
        a=a_voigt
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
        vH=vH,
        a=a_voigt
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
            vH=vH,
            a=a_voigt
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
            vH=vH,
            a=a_voigt
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
Hu = 1.0 # depends on B
#vH = 0.0
vH = 1.3996e6 * B / default_Delta_nu_D

print("vH = ", vH)
print("a_voigt = ", a_voigt)

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
    #print("\nFrequency x = ", x)
    #print("rho2 real:", np.real(rho2))
    #print("rho2 imag:", np.imag(rho2))
    epsI = 0.0+0j
    epsQ = 0.0+0j
    epsU = 0.0+0j
    epsV = 0.0+0j
    # K=0 blocks
    Phi00 = Phi_generalized(np.array([x]), K=0, Kp=0, Q=0, vH=vH, a=a_voigt)[0]
    epsI += Phi00 * T(0,0,0,theta_obs,chi_obs,gamma_obs) * J00

    Phi01 = Phi_generalized(np.array([x]), K=0, Kp=1, Q=0, vH=vH, a=a_voigt)[0]
    epsV += Phi01 * T(3,1,0, theta_obs, chi_obs, gamma_obs) * J00

    Phi02 = Phi_generalized(np.array([x]), K=0, Kp=2, Q=0, vH=vH, a=a_voigt)[0]
    epsI += Phi02 * T(0,2,0,theta_obs,chi_obs,gamma_obs) * J00
    epsQ += Phi02 * T(1,2,0,theta_obs,chi_obs,gamma_obs) * J00
    epsU += Phi02 * T(2,2,0,theta_obs,chi_obs,gamma_obs) * J00

    V21 = 0+0j

    # K=2 blocks
    for Q in [-2,-1,0,1,2]:
        phase = (-1)**Q
        rhoQ = np.conj(rho2[idx(-Q)])
        #phase = 1
        #rhoQ = rho2[idx(Q)]

        Phi20 = Phi_generalized(np.array([x]), K=2, Kp=0, Q=Q, vH=vH, a=a_voigt)[0]
        epsI += phase * Phi20 * T(0,0,0,theta_obs,chi_obs,gamma_obs) * rhoQ

        Phi21 = Phi_generalized(np.array([x]), K=2, Kp=1, Q=Q, vH=vH, a=a_voigt)[0]
    
        # ---------------------------------------
        # DEBUG ONLY AT ONE FREQUENCY
        # ---------------------------------------
        if abs(x - 0.5) < 1e-10:
            print(f"\nFrequency x={x}")
            term = (
                phase
                * Phi21
                * T(3,1,Q,theta_obs,chi_obs,gamma_obs)
                * rhoQ
            )

            print(f"\nQ = {Q}")
            print(f"Phi21 = {Phi21}")
            print(f"T31    = {T(3,1,Q,theta_obs,chi_obs,gamma_obs)}")
            print(f"rhoQ   = {rhoQ}")
            print(f"Product = {rhoQ * Phi21}")
            print(f"term   = {term}")
            print(f"term.real   = {term.real}")
            print(f"term.imag   = {term.imag}")
        epsI += phase * Phi21 * T(0,1,Q,theta_obs,chi_obs,gamma_obs) * rhoQ
        epsQ += phase * Phi21 * T(1,1,Q,theta_obs,chi_obs,gamma_obs) * rhoQ
        epsU += phase * Phi21 * T(2,1,Q,theta_obs,chi_obs,gamma_obs) * rhoQ
        epsV += phase * Phi21 * T(3,1,Q,theta_obs,chi_obs,gamma_obs) * rhoQ
        if abs(x - 0.5) < 1e-10:
            V21 += phase * Phi21 * T(3,1,Q,theta_obs,chi_obs,gamma_obs) * rhoQ
            print("V21 contribution for Q =", Q, "is", V21)

        Phi22 = Phi_generalized(np.array([x]), K=2, Kp=2, Q=Q, vH=vH, a=a_voigt)[0]
        epsI += phase * Phi22 * T(0,2,Q,theta_obs,chi_obs,gamma_obs) * rhoQ
        epsQ += phase * Phi22 * T(1,2,Q,theta_obs,chi_obs,gamma_obs) * rhoQ
        epsU += phase * Phi22 * T(2,2,Q,theta_obs,chi_obs,gamma_obs) * rhoQ
            
    I_prof[ix] = (np.real(epsI)*np.sqrt(np.pi))
    Q_prof[ix] = (np.real(epsQ)*np.sqrt(np.pi))
    U_prof[ix] = (np.real(epsU)*np.sqrt(np.pi))
    V_prof[ix] = (np.real(epsV)*np.sqrt(np.pi))
    #print(f"Computed Stokes profiles at x={x:.3f}: I={I_prof[ix]:.6f}, Q={Q_prof[ix]:.6f}, U={U_prof[ix]:.6f}, V={V_prof[ix]}")

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
plt.savefig("Stokes_try_Hu{}_gamma{}_theta{}_vH{}.png".format(Hu, np.degrees(gamma_obs), np.degrees(theta_obs), vH), dpi = 300)

fig = plt.figure(figsize=(8,6))
plt.plot(xgrid, Q_prof/I_prof, label="P_Q")
plt.plot(xgrid, U_prof/I_prof, label="P_U")
plt.plot(xgrid, V_prof/I_prof, label="P_V")
plt.xlabel("x")
plt.ylabel("P")
plt.title("P vs x for different Stokes parameters")
plt.legend()
plt.savefig("Fractional_polarization_Hu{}_gamma{}_theta{}_vH{}.png".format(Hu, np.degrees(gamma_obs), np.degrees(theta_obs), vH), dpi = 300)

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
    vH=1.0,
    a=a_voigt
)
Phi_Q1 = Phi_generalized(
    xgrid,
    K=2,
    Kp=1,
    Q=1,
    vH=1.0,
    a=a_voigt
)
Phi_Q2 = Phi_generalized(
    xgrid,
    K=2,
    Kp=1,
    Q=2,
    vH=1.0,
    a=a_voigt
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
            vH,
            a=a_voigt
        )[0]

        print(K,Kp,val)

for Q in [-2,-1,0,1,2]:
    print(Q,
          Phi_generalized(
             np.array([0]),
             2,
             1,
             Q,
             vH,
             a=a_voigt
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
                vH=1.0,
                a=a_voigt
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
    P = Phi_generalized(x, K=2, Kp=1, Q=0, vH=1.0, a=a_voigt)  # run before/after edit
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

        Phi00 = Phi_generalized(np.array([x]), K=0, Kp=0, Q=0, vH=vH, a=a_voigt)[0]
        epsI += Phi00 * T_choice(0, 0, 0) * J00

        Phi01 = Phi_generalized(np.array([x]), K=0, Kp=1, Q=0, vH=vH, a=a_voigt)[0]
        epsV += Phi01 * T(3,1,0, theta_obs, chi_obs, gamma_obs) * J00

        Phi02 = Phi_generalized(np.array([x]), K=0, Kp=2, Q=0, vH=vH, a=a_voigt)[0]
        epsI += Phi02 * T_choice(0, 2, 0) * J00
        epsQ += Phi02 * T_choice(1, 2, 0) * J00
        epsU += Phi02 * T_choice(2, 2, 0) * J00

        for Q in [-2, -1, 0, 1, 2]:
            phase = (-1) ** Q
            rhoQ = np.conj(rho2[idx(-Q)])

            Phi20 = Phi_generalized(np.array([x]), K=2, Kp=0, Q=Q, vH=vH, a=a_voigt)[0]
            epsI += phase * Phi20 * T_choice(0, 0, Q) * rhoQ

            Phi21 = Phi_generalized(np.array([x]), K=2, Kp=1, Q=Q, vH=vH, a=a_voigt)[0]
            epsI += phase * Phi21 * T_choice(0, 1, Q) * rhoQ
            epsQ += phase * Phi21 * T_choice(1, 1, Q) * rhoQ
            epsU += phase * Phi21 * T_choice(2, 1, Q) * rhoQ
            epsV += phase * Phi21 * T(3, 1, Q, theta_obs, chi_obs, gamma_obs) * rhoQ

            Phi22 = Phi_generalized(np.array([x]), K=2, Kp=2, Q=Q, vH=vH, a=a_voigt)[0]
            epsI += phase * Phi22 * T_choice(0, 2, Q) * rhoQ
            epsQ += phase * Phi22 * T_choice(1, 2, Q) * rhoQ
            epsU += phase * Phi22 * T_choice(2, 2, Q) * rhoQ

        I_prof[ix] = np.real(epsI) * np.sqrt(np.pi)
        Q_prof[ix] = np.real(epsQ) * np.sqrt(np.pi)
        U_prof[ix] = np.real(epsU) * np.sqrt(np.pi)
        V_prof[ix] = np.real(epsV) * np.sqrt(np.pi)

    return I_prof, Q_prof, U_prof, V_prof


compare_T_vs_T_book(theta_obs, chi_obs)

I_T, Q_T, U_T, V_T = compute_stokes_profiles(
    theta_obs, chi_obs, gamma_obs, use_T_book=False
)
I_B, Q_B, U_B, V_B = compute_stokes_profiles(
    theta_obs, chi_obs, gamma_obs, use_T_book=True
)

fig, ax = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle("Stokes profiles comparison: T vs T_book for Hu = {}, theta_B = {}, chi_B = {}, theta_obs = {}, chi_obs = {}".format(Hu, np.degrees(theta_B), chi_B, np.degrees(theta_obs), np.degrees(chi_obs)))
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
plt.savefig("Stokes_compare_T_Tbook_Hu{}.png".format(Hu), dpi=300)
print("Saved: Stokes_compare_T_Tbook_Hu{}.png".format(Hu))

# Comparison
def compare_appendix_phi():
    print("\n=== Appendix vs Phi_generalized comparison ===")
    for K in [0, 1, 2]:
        for Kp in [0, 1, 2]:
            for Q in [-2, -1, 0, 1, 2]:
                if abs(Q) > min(K, Kp):
                    continue
                P_gen = Phi_generalized(xgrid, K, Kp, Q, vH, a=a_voigt)
                P_app = Phi_appendix(xgrid, K, Kp, Q, vH)
                diff = np.max(np.abs(P_gen - P_app))
                print(f"K={K} K'={Kp} Q={Q} max|gen-app| = {diff:.3e}")

    # plot one representative case
    K, Kp, Q = 2, 2, 0
    P_gen = Phi_generalized(xgrid, K, Kp, Q, vH)
    P_app = Phi_appendix(xgrid, K, Kp, Q, vH)

    plt.figure(figsize=(8, 5))
    plt.plot(xgrid, np.real(P_gen), 'r-', label='Phi_generalized Re')
    plt.plot(xgrid, np.real(P_app), 'b--', label='Phi_appendix Re')
    plt.plot(xgrid, np.imag(P_gen), 'g-', label='Phi_generalized Im')
    plt.plot(xgrid, np.imag(P_app), 'k--', label='Phi_appendix Im')
    plt.title(f"Appendix check: K={K}, K'={Kp}, Q={Q}")
    plt.legend()
    plt.tight_layout()
    plt.savefig("Phi_appendix_vs_generalized.png", dpi=300)

#compare_appendix_phi()

def compute_stokes_profiles_appendix(
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

        Phi00 = Phi_appendix(np.array([x]), K=0, Kp=0, Q=0, vH=vH, a=a_voigt)[0]
        epsI += Phi00 * T_choice(0, 0, 0) * J00

        Phi01 = Phi_generalized(np.array([x]), K=0, Kp=1, Q=0, vH=vH, a=a_voigt)[0]
        epsV += Phi01 * T(3,1,0, theta_obs, chi_obs, gamma_obs) * J00

        Phi02 = Phi_appendix(np.array([x]), K=0, Kp=2, Q=0, vH=vH, a=a_voigt)[0]
        epsI += Phi02 * T_choice(0, 2, 0) * J00
        epsQ += Phi02 * T_choice(1, 2, 0) * J00
        epsU += Phi02 * T_choice(2, 2, 0) * J00

        for Q in [-2, -1, 0, 1, 2]:
            phase = (-1) ** Q
            rhoQ = np.conj(rho2[idx(-Q)])

            Phi20 = Phi_appendix(np.array([x]), K=2, Kp=0, Q=Q, vH=vH, a=a_voigt)[0]
            epsI += phase * Phi20 * T_choice(0, 0, Q) * rhoQ

            Phi21 = Phi_appendix(np.array([x]), K=2, Kp=1, Q=Q, vH=vH, a=a_voigt)[0]
            epsI += phase * Phi21 * T_choice(0, 1, Q) * rhoQ
            epsQ += phase * Phi21 * T_choice(1, 1, Q) * rhoQ
            epsU += phase * Phi21 * T_choice(2, 1, Q) * rhoQ
            epsV += phase * Phi21 * T(3, 1, Q, theta_obs, chi_obs, gamma_obs) * rhoQ

            Phi22 = Phi_appendix(np.array([x]), K=2, Kp=2, Q=Q, vH=vH, a=a_voigt)[0]
            epsI += phase * Phi22 * T_choice(0, 2, Q) * rhoQ
            epsQ += phase * Phi22 * T_choice(1, 2, Q) * rhoQ
            epsU += phase * Phi22 * T_choice(2, 2, Q) * rhoQ

        I_prof[ix] = np.real(epsI) * np.sqrt(np.pi)
        Q_prof[ix] = np.real(epsQ) * np.sqrt(np.pi)
        U_prof[ix] = np.real(epsU) * np.sqrt(np.pi)
        V_prof[ix] = np.real(epsV) * np.sqrt(np.pi)

    return I_prof, Q_prof, U_prof, V_prof


compare_T_vs_T_book(theta_obs, chi_obs)

I_T, Q_T, U_T, V_T = compute_stokes_profiles_appendix(
    theta_obs, chi_obs, gamma_obs, use_T_book=False
)
I_B, Q_B, U_B, V_B = compute_stokes_profiles_appendix(
    theta_obs, chi_obs, gamma_obs, use_T_book=True
)

fig, ax = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle("Stokes profiles comparison: T vs T_book (Appendix) for Hu = {}, theta_B = {}, chi_B = {}, theta_obs = {}, chi_obs = {}".format(Hu, np.degrees(theta_B), chi_B, np.degrees(theta_obs), np.degrees(chi_obs)))
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
plt.savefig("Stokes_compare_T_Tbook_Hu{}_phi_appendix.png".format(Hu), dpi=300)
print("Saved: Stokes_compare_T_Tbook_Hu{}_phi_appendix.png".format(Hu))

# Hu = 0.0 for testing
def compute_stokes_profiles_phi(
        theta_obs,
        chi_obs,
        gamma_obs,
        use_appendix=False):

    Jarr = Jrad_to_array(Jrad_0)
    rho2 = apply_hanle(Jarr, Hu, theta_B, chi_B)

    I_prof = np.zeros_like(xgrid)
    Q_prof = np.zeros_like(xgrid)
    U_prof = np.zeros_like(xgrid)
    V_prof = np.zeros_like(xgrid)

    def Phi_choice(K, Kp, Q, x):
        if use_appendix:
            return Phi_appendix(np.array([x]), K=K, Kp=Kp, Q=Q, vH=vH, a=a_voigt)[0]
        return Phi_generalized(np.array([x]), K=K, Kp=Kp, Q=Q, vH=vH, a=a_voigt)[0]

    for ix, x in enumerate(xgrid):
        J00 = Jrad_0[(0, 0)]
        epsI = 0.0 + 0j
        epsQ = 0.0 + 0j
        epsU = 0.0 + 0j
        epsV = 0.0 + 0j

        Phi00 = Phi_choice(0, 0, 0, x)
        epsI += Phi00 * T(0, 0, 0, theta_obs, chi_obs, gamma_obs) * J00

        Phi01 = Phi_choice(0, 1, 0, x)
        epsV += Phi01 * T(3, 1, 0, theta_obs, chi_obs, gamma_obs) * J00

        Phi02 = Phi_choice(0, 2, 0, x)
        epsI += Phi02 * T(0, 2, 0, theta_obs, chi_obs, gamma_obs) * J00
        epsQ += Phi02 * T(1, 2, 0, theta_obs, chi_obs, gamma_obs) * J00
        epsU += Phi02 * T(2, 2, 0, theta_obs, chi_obs, gamma_obs) * J00

        for Q in [-2, -1, 0, 1, 2]:
            phase = (-1) ** Q
            rhoQ = np.conj(rho2[idx(-Q)])

            Phi20 = Phi_choice(2, 0, Q, x)
            epsI += phase * Phi20 * T(0, 0, Q, theta_obs, chi_obs, gamma_obs) * rhoQ

            Phi21 = Phi_choice(2, 1, Q, x)
            epsI += phase * Phi21 * T(0, 1, Q, theta_obs, chi_obs, gamma_obs) * rhoQ
            epsQ += phase * Phi21 * T(1, 1, Q, theta_obs, chi_obs, gamma_obs) * rhoQ
            epsU += phase * Phi21 * T(2, 1, Q, theta_obs, chi_obs, gamma_obs) * rhoQ
            epsV += phase * Phi21 * T(3, 1, Q, theta_obs, chi_obs, gamma_obs) * rhoQ

            Phi22 = Phi_choice(2, 2, Q, x)
            epsI += phase * Phi22 * T(0, 2, Q, theta_obs, chi_obs, gamma_obs) * rhoQ
            epsQ += phase * Phi22 * T(1, 2, Q, theta_obs, chi_obs, gamma_obs) * rhoQ
            epsU += phase * Phi22 * T(2, 2, Q, theta_obs, chi_obs, gamma_obs) * rhoQ

        I_prof[ix] = np.real(epsI) * np.sqrt(np.pi)
        Q_prof[ix] = np.real(epsQ) * np.sqrt(np.pi)
        U_prof[ix] = np.real(epsU) * np.sqrt(np.pi)
        V_prof[ix] = np.real(epsV) * np.sqrt(np.pi)

    return I_prof, Q_prof, U_prof, V_prof


I_gen, Q_gen, U_gen, V_gen = compute_stokes_profiles_phi(
    theta_obs, chi_obs, gamma_obs, use_appendix=False
)
I_app, Q_app, U_app, V_app = compute_stokes_profiles_phi(
    theta_obs, chi_obs, gamma_obs, use_appendix=True
)

fig, ax = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle("Stokes profiles comparison: Phi_generalized vs Phi_appendix for Hu = {}, theta_B = {}, chi_B = {}, theta_obs = {}, chi_obs = {}".format(Hu, np.degrees(theta_B), chi_B, np.degrees(theta_obs), np.degrees(chi_obs)))
for row, label, data_gen, data_app in [
        (0, "I", I_gen, I_app),
        (1, "Q", Q_gen, Q_app),
        (2, "U", U_gen, U_app),
        (3, "V", V_gen, V_app),
]:
    r = row // 2
    c = row % 2
    ax[r, c].plot(xgrid, data_gen, "r-", label="Phi_generalized")
    ax[r, c].plot(xgrid, data_app, "b--", label="Phi_appendix")
    ax[r, c].set_title(label)
    ax[r, c].legend()

plt.tight_layout()
plt.savefig("Stokes_compare_Phi_generalized_vs_appendix_Hu{}.png".format(Hu), dpi=300)
print("Saved: Stokes_compare_Phi_generalized_vs_appendix_Hu{}.png".format(Hu))

x = np.linspace(-5, 5, 501)
a_plot = 1.0
print(f"Damping parameter a = {a_voigt:.3e}")
phi_no_damp = phi_complex(x, a=0.0)
phi_damp = phi_complex(x, a=a_plot)
plt.figure(figsize=(8, 5))
plt.plot(x, np.real(phi_no_damp), 'r-', label='Re, a=0')
plt.plot(x, np.real(phi_damp), 'r--', label=f'Re, a={a_plot:.1e}')
plt.plot(x, np.imag(phi_no_damp), 'b-', label='Im, a=0')
plt.plot(x, np.imag(phi_damp), 'b--', label=f'Im, a={a_plot:.1e}')
plt.legend()
plt.savefig("phi_complex_damping_comparison.png", dpi=300)
print("Saved: phi_complex_damping_comparison.png")

def compare_k01_v_block(x_value=0.0):
    """Compare the Stokes-V emissivity with and without the K=0, K'=1 block."""
    Jarr = Jrad_to_array(Jrad_0)
    rho2 = apply_hanle(Jarr, Hu, theta_B, chi_B)
    J00 = Jrad_0[(0, 0)]

    epsV_without = 0.0 + 0.0j
    epsV_with = 0.0 + 0.0j

    Phi00 = Phi_generalized(np.array([x_value]), K=0, Kp=0, Q=0, vH=vH, a=a_voigt)[0]
    epsV_without += Phi00 * T(0, 0, 0, theta_obs, chi_obs, gamma_obs) * J00
    epsV_with += Phi00 * T(0, 0, 0, theta_obs, chi_obs, gamma_obs) * J00

    Phi02 = Phi_generalized(np.array([x_value]), K=0, Kp=2, Q=0, vH=vH, a=a_voigt)[0]
    epsV_without += 0.0j
    epsV_with += 0.0j

    Phi01 = Phi_generalized(np.array([x_value]), K=0, Kp=1, Q=0, vH=vH, a=a_voigt)[0]
    epsV_with += Phi01 * T(3, 1, 0, theta_obs, chi_obs, gamma_obs) * J00

    for Q in [-2, -1, 0, 1, 2]:
        phase = (-1) ** Q
        rhoQ = np.conj(rho2[idx(-Q)])

        Phi21 = Phi_generalized(np.array([x_value]), K=2, Kp=1, Q=Q, vH=vH, a=a_voigt)[0]
        epsV_without += phase * Phi21 * T(3, 1, Q, theta_obs, chi_obs, gamma_obs) * rhoQ
        epsV_with += phase * Phi21 * T(3, 1, Q, theta_obs, chi_obs, gamma_obs) * rhoQ

    return epsV_without, epsV_with


V_without, V_with = compare_k01_v_block(x_value=np.linspace(-5, 5, 501))  # x=0.0
print("V(x=0) without K=0,K'=1 block:", V_without)
print("V(x=0) with    K=0,K'=1 block:", V_with)
print("delta V(x=0):", V_with - V_without)

# Symmetry
print("\n=== Symmetry check for rho(Q) ===")
for Q in [-2, -1, 0, 1, 2]:
    lhs = rho[idx(Q)]
    rhs = (-1)**Q * (rho[idx(-Q)])
    print(f"Q={Q}: rho={lhs}, (-1)^Q * conj(rho[-Q])={rhs}, diff={lhs-rhs}")

print("\nTensor values for K'=2, i=1 (Stokes Q) and i=2 (Stokes U):")
for Q in [-2, -1, 0, 1, 2]:
    T_Q = T(1, 2, Q, theta_obs, chi_obs, gamma_obs) # Stokes Q
    T_U = T(2, 2, Q, theta_obs, chi_obs, gamma_obs) # Stokes U
    print(f"Q={Q}: T_Q={T_Q}, T_U={T_U}")

for Q in [-1,0,1]:
    Phi = np.array([
        Phi_generalized(np.array([x]),2,1,Q,vH,a=a_voigt)[0]
        for x in xgrid
    ])

    plt.figure()
    plt.plot(xgrid, Phi.real, label="Real")
    plt.plot(xgrid, Phi.imag, label="Imag")
    plt.title(f"K=2, K'=1, Q={Q}")
    plt.legend()
    plt.savefig(f"Phi_K2_Kp1_Q{Q}.png", dpi=300)

fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharex=True, sharey=True)

for ax, Q in zip(axes, [-1, 0, 1]):
    Phi = np.array([
        Phi_generalized(np.array([x]), K=2, Kp=1, Q=Q, vH=vH, a = a_voigt)[0]
        for x in xgrid
    ])

    ax.plot(xgrid, Phi.real, label="Real")
    ax.plot(xgrid, Phi.imag, label="Imag")
    ax.set_title(f"$Q={Q}$")
    ax.grid(True)

axes[0].set_ylabel(r"$\Phi^{21}_Q$")
for ax in axes:
    ax.set_xlabel(r"$x=(\nu-\nu_0)/\Delta\nu_D$")

# Show legend only once
axes[0].legend()

plt.tight_layout()
plt.savefig("Phi_K2_Kp1_Q_all.png", dpi=300)


fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharex=True, sharey=True)

for ax, Q in zip(axes, [-1, 0, 1]):
    Phi = np.array([
        Phi_appendix(np.array([x]), K=2, Kp=1, Q=Q, vH=vH, a = a_voigt)[0]
        for x in xgrid
    ])

    ax.plot(xgrid, Phi.real, label="Real")
    ax.plot(xgrid, Phi.imag, label="Imag")
    ax.set_title(f"$Q={Q}$")
    ax.grid(True)

axes[0].set_ylabel(r"$\Phi^{21}_Q$")
for ax in axes:
    ax.set_xlabel(r"$x=(\nu-\nu_0)/\Delta\nu_D$")

# Show legend only once
axes[0].legend()

plt.tight_layout()
plt.savefig("Phi_K2_Kp1_Q_all_appendix.png", dpi=300)


print("i = 3")
for Q in [-1,0,1]:
    print(Q, T(3,1,Q,theta_obs,chi_obs,np.pi/2))

print("J_arr versus rho2:")
for Q in [-2,-1,0,1,2]:
    print(Q, rho2[idx(Q)])
Jarr = Jrad_to_array(Jrad_0)
print("Jarr:", Jarr)
print("Jarr/rho2", Jrad_to_array(Jrad_0)/rho2)

print("cancellation between Q=±1")
for Q in [-1,1]:

    Phi = Phi_generalized(
        np.array(np.array([0.5])),
        K=2,
        Kp=1,
        Q=Q,
        vH=vH,
        a=a_voigt
    )[0]

    contrib = (
        (-1)**Q
        * Phi
        * T(3,1,Q,theta_obs,chi_obs,np.pi/2)
        * rho2[idx(Q)]
    )

    print("Q = ", Q)
    print("Phi =", Phi)
    print("T   =", T(3,1,Q,theta_obs,chi_obs,np.pi/2))
    print("rho =", rho2[idx(Q)])
    print("contrib =", contrib)

print("Profile values at x=0.5 for K=2, K'=1, Q=±1:")
for Q in [-1,1]:
    print(f"Q = {Q}")

    Phi = Phi_generalized(
        np.array(np.array([0.5])),
        K=2,
        Kp=1,
        Q=Q,
        vH=vH,
        a=a_voigt
    )[0]

    print(f"Phi = {Phi}")

for Q in [-1,0,1]:
    for x in [-1.0, -0.5, 0.5, 1.0]:
        print(f"x={x}, Q={Q}, Phi={Phi_generalized(np.array([x]),2,1,Q,vH,a=a_voigt)[0]}")

print("Direct division by 1+i*Q*Hu")
print(np.shape(Jrad_0))
for ix, x in enumerate(xgrid):
    Jarr = Jrad_to_array(Jrad_0)
    J00 = Jrad_0[(0,0)]
    rho2 = apply_hanle(Jarr, Hu, theta_B, chi_B)
    epsI = 0.0+0j
    epsQ = 0.0+0j
    epsU = 0.0+0j
    epsV = 0.0+0j
    # K=0 blocks
    Phi00 = Phi_generalized(np.array([x]), K=0, Kp=0, Q=0, vH=vH, a=a_voigt)[0]
    epsI += Phi00 * T(0,0,0,theta_obs,chi_obs,gamma_obs) * J00

    Phi01 = Phi_generalized(np.array([x]), K=0, Kp=1, Q=0, vH=vH, a=a_voigt)[0]
    epsV += Phi01 * T(3,1,0, theta_obs, chi_obs, gamma_obs) * J00

    Phi02 = Phi_generalized(np.array([x]), K=0, Kp=2, Q=0, vH=vH, a=a_voigt)[0]
    epsI += Phi02 * T(0,2,0,theta_obs,chi_obs,gamma_obs) * J00
    epsQ += Phi02 * T(1,2,0,theta_obs,chi_obs,gamma_obs) * J00
    epsU += Phi02 * T(2,2,0,theta_obs,chi_obs,gamma_obs) * J00

    # K=2 blocks
    for Q in [-2,-1,0,1,2]:
        #phase = (-1)**Q
        rhoQ = np.conj(rho2[idx(-Q)])
        phase = 1
        #rhoQ = rho2[idx(Q)]
        JQ = Jrad_0[(2, -Q)]
        hanle = (-1)**Q * JQ / (1 - 1j*Q*Hu)

        Phi20 = Phi_generalized(np.array([x]), K=2, Kp=0, Q=Q, vH=vH, a=a_voigt)[0]
        epsI += phase * Phi20 * T(0,0,0,theta_obs,chi_obs,gamma_obs) * hanle

        Phi21 = Phi_generalized(np.array([x]), K=2, Kp=1, Q=Q, vH=vH, a=a_voigt)[0]
    
        # ---------------------------------------
        # DEBUG ONLY AT ONE FREQUENCY
        # ---------------------------------------
        if abs(x - 0.5) < 1e-10:
            print(f"\nFrequency x={x}")
            term = (
                phase
                * Phi21
                * T(3,1,Q,theta_obs,chi_obs,gamma_obs)
                * rhoQ
            )

            print(f"\nQ = {Q}")
            print(f"Phi21 = {Phi21}")
            print(f"T31    = {T(3,1,Q,theta_obs,chi_obs,gamma_obs)}")
            print(f"rhoQ   = {rhoQ}")
            print(f"term.real   = {term.real}")
            print(f"term.imag   = {term.imag}")
        epsI += phase * Phi21 * T(0,1,Q,theta_obs,chi_obs,gamma_obs) * hanle
        epsQ += phase * Phi21 * T(1,1,Q,theta_obs,chi_obs,gamma_obs) * hanle
        epsU += phase * Phi21 * T(2,1,Q,theta_obs,chi_obs,gamma_obs) * hanle
        epsV += phase * Phi21 * T(3,1,Q,theta_obs,chi_obs,gamma_obs) * hanle

        Phi22 = Phi_generalized(np.array([x]), K=2, Kp=2, Q=Q, vH=vH, a=a_voigt)[0]
        epsI += phase * Phi22 * T(0,2,Q,theta_obs,chi_obs,gamma_obs) * hanle
        epsQ += phase * Phi22 * T(1,2,Q,theta_obs,chi_obs,gamma_obs) * hanle
        epsU += phase * Phi22 * T(2,2,Q,theta_obs,chi_obs,gamma_obs) * hanle
            
    I_prof[ix] = (np.real(epsI)*np.sqrt(np.pi))
    Q_prof[ix] = (np.real(epsQ)*np.sqrt(np.pi))
    U_prof[ix] = (np.real(epsU)*np.sqrt(np.pi))
    V_prof[ix] = (np.real(epsV)*np.sqrt(np.pi))
    #print(f"Computed Stokes profiles at x={x:.3f}: I={I_prof[ix]:.6f}, Q={Q_prof[ix]:.6f}, U={U_prof[ix]:.6f}, V={V_prof[ix]}")

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
plt.savefig("Stokes_direct_Hu{}_gamma{}_theta{}.png".format(Hu, np.degrees(gamma_obs), np.degrees(theta_obs)), dpi = 300)

Phi20_array = np.array([
    Phi_generalized(np.array([xx]),2,0,1,vH,a_voigt)[0]
    for xx in xgrid
])


Phi21_array = np.array([
    Phi_generalized(np.array([xx]),2,1,1,vH,a_voigt)[0]
    for xx in xgrid
])

Phi22_array = np.array([
    Phi_generalized(np.array([xx]),2,2,1,vH,a_voigt)[0]
    for xx in xgrid])
 
print("Max values of Phi20, Phi21, Phi22 for Q=1:")
print(f"Phi20: {np.max(np.abs(Phi20_array))}")
print(f"Phi21: {np.max(np.abs(Phi21_array))}")
print(f"Phi22: {np.max(np.abs(Phi22_array))}")

# Freq

Phi20_array = np.array([
    Phi_generalized(np.array([xx]), K=2, Kp=0, Q=1,
                    vH=vH, a=a_voigt)[0]
    for xx in xgrid
])

Phi21_array = np.array([
    Phi_generalized(np.array([xx]), K=2, Kp=1, Q=1,
                    vH=vH, a=a_voigt)[0]
    for xx in xgrid
])

Phi22_array = np.array([
    Phi_generalized(np.array([xx]), K=2, Kp=2, Q=1,
                    vH=vH, a=a_voigt)[0]
    for xx in xgrid
])

plt.figure(figsize=(8,6))
plt.plot(xgrid, np.real(Phi20_array), label="Phi20")
plt.plot(xgrid, np.real(Phi21_array), label="Phi21")
plt.plot(xgrid, np.real(Phi22_array), label="Phi22")
plt.legend()
plt.savefig("Phi_K2_Q1_real.png", dpi=300)

plt.figure(figsize=(8,6))

plt.plot(
    xgrid,
    np.real(Phi20_array),
    label="Phi20"
)

plt.plot(
    xgrid,
    np.real(Phi21_array)/np.max(np.abs(Phi21_array)),
    label="Phi21"
)

plt.plot(
    xgrid,
    np.real(Phi22_array)/np.max(np.abs(Phi22_array)),
    label="Phi22"
)

plt.legend()
plt.savefig("Phi_K2_Q1_real_normalized.png", dpi=300)

Phi20_0array = np.array([
    Phi_generalized(np.array([xx]), K=2, Kp=0, Q=0,
                    vH=vH, a=a_voigt)[0]
    for xx in xgrid
])

Phi21_0array = np.array([
    Phi_generalized(np.array([xx]), K=2, Kp=1, Q=0,
                    vH=vH, a=a_voigt)[0]
    for xx in xgrid
])

Phi22_0array = np.array([
    Phi_generalized(np.array([xx]), K=2, Kp=2, Q=0,
                    vH=vH, a=a_voigt)[0]
    for xx in xgrid
])

plt.figure(figsize=(8,6))
plt.plot(xgrid, np.real(Phi20_0array), label="Phi20_0")
plt.plot(xgrid, np.real(Phi21_0array), label="Phi21_0")
plt.plot(xgrid, np.real(Phi22_0array), label="Phi22_0")
plt.legend()
plt.savefig("Phi_K2_Q0_real.png", dpi=300)

plt.figure(figsize=(8,6))

plt.plot(
    xgrid,
    np.real(Phi20_0array),
    label="Phi20_0"
)

plt.plot(
    xgrid,
    np.real(Phi21_0array)/np.max(np.abs(Phi21_0array)),
    label="Phi21_0"
)

plt.plot(
    xgrid,
    np.real(Phi22_0array)/np.max(np.abs(Phi22_0array)),
    label="Phi22_0"
)

plt.legend()
plt.savefig("Phi_K2_Q0_real_normalized.png", dpi=300)

from scipy.special import wofz

xgrid = np.linspace(-5,5,101)

phi_conv = np.zeros_like(xgrid,dtype=complex)
phi_wofz = np.zeros_like(xgrid,dtype=complex)

for i,x in enumerate(xgrid):

    phi_conv[i] = Phi_convolved(
        x,
        A_ul,
        default_Delta_nu_D
    )

    phi_wofz[i] = wofz(x+1j*a_voigt)/np.sqrt(np.pi)

print(phi_conv[50], phi_wofz[50])


plt.figure(figsize=(8,5))

plt.plot(xgrid,
         np.real(phi_conv),
         label="Convolution")

plt.plot(xgrid,
         np.real(phi_wofz),
         "--",
         label="wofz")

plt.legend()
plt.xlabel("x")
plt.ylabel("Real part")
plt.grid()
plt.savefig("phi_convolution_vs_wofz_real.png", dpi=300)

plt.figure(figsize=(8,5))

plt.plot(xgrid,
         np.imag(phi_conv),
         label="Convolution")

plt.plot(xgrid,
         np.imag(phi_wofz),
         "--",
         label="wofz")

plt.legend()
plt.xlabel("x")
plt.ylabel("Imaginary part")
plt.grid()
plt.savefig("phi_convolution_vs_wofz_imaginary.png", dpi=300)

diff = phi_conv - phi_wofz

print("Maximum real error:",
      np.max(np.abs(diff.real)))

print("Maximum imaginary error:",
      np.max(np.abs(diff.imag)))