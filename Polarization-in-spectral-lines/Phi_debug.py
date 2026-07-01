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
    epsI += Phi00 * T(0,0,0,theta_obs,chi_obs,np.pi/2) * J00

    Phi02 = Phi_generalized(np.array([x]), K=0, Kp=2, Q=0, vH=vH)[0]
    epsI += Phi02 * T(0,2,0,theta_obs,chi_obs,np.pi/2) * J00
    epsQ += Phi02 * T(1,2,0,theta_obs,chi_obs,np.pi/2) * J00
    epsU += Phi02 * T(2,2,0,theta_obs,chi_obs,np.pi/2) * J00

    # K=2 blocks
    for Q in [-2,-1,0,1,2]:
        phase = (-1)**Q
        rhoQ = rho2[idx(-Q)]

        Phi20 = Phi_generalized(np.array([x]), K=2, Kp=0, Q=Q, vH=vH)[0]
        epsI += phase * Phi20 * T(0,0,0,theta_obs,chi_obs,np.pi/2) * rhoQ

        Phi21 = Phi_generalized(np.array([x]), K=2, Kp=1, Q=Q, vH=vH)[0]
        epsI += phase * Phi21 * T(0,1,Q,theta_obs,chi_obs,np.pi/2) * rhoQ
        epsQ += phase * Phi21 * T(1,1,Q,theta_obs,chi_obs,np.pi/2) * rhoQ
        epsU += phase * Phi21 * T(2,1,Q,theta_obs,chi_obs,np.pi/2) * rhoQ
        epsV += phase * Phi21 * T(3,1,Q,theta_obs,chi_obs,np.pi/2) * rhoQ

        Phi22 = Phi_generalized(np.array([x]), K=2, Kp=2, Q=Q, vH=vH)[0]
        epsI += phase * Phi22 * T(0,2,Q,theta_obs,chi_obs,np.pi/2) * rhoQ
        epsQ += phase * Phi22 * T(1,2,Q,theta_obs,chi_obs,np.pi/2) * rhoQ
        epsU += phase * Phi22 * T(2,2,Q,theta_obs,chi_obs,np.pi/2) * rhoQ
            
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
plt.savefig("Stokes_try.png", dpi = 300)

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

