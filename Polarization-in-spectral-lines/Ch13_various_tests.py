import numpy as np
from sympy.physics.wigner import wigner_d_small
import sys
import os
import matplotlib.pyplot as plt
# Add the directory containing functions_prt.py to the Python path

script_dir = os.path.abspath("/home/Code/NLTE-polarized-radiation")
# script_dir = os.path.abspath("/home/teodor/Documents/Code/NLTE-polarized-radiation")
sys.path.append(script_dir)

from functions_prt import wigner_d2, wigner_D2
from Ch13_short import anisotropy_w2, anisotropy

sqrt2 = np.sqrt(2.0)
sqrt3 = np.sqrt(3.0)

def T(i, K, Q, theta, chi):
    """
    Irreducible spherical tensor T^K_Q(i,Omega)
    following Landi Degl'Innocenti & Landolfi Table 5.6

    i = 0,1,2,3  -> I,Q,U,V
    """

    # negative-Q relation
    if Q < 0:
        return (-1)**Q * np.conj(
            T(i,K,-Q,theta,chi)
        )

    ct = np.cos(theta)
    st = np.sin(theta)

    c2 = np.cos(2*chi)
    s2 = np.sin(2*chi)

    ex1 = np.exp(1j*chi)
    ex2 = np.exp(2j*chi)

    # ------------------
    # STOKES I
    # ------------------

    if i == 0:

        if K == 0 and Q == 0:
            return 1.0

        if K == 2 and Q == 0:
            return (3*ct**2 - 1)/(2*sqrt2)

        if K == 2 and Q == 1:
            return -sqrt3/2 * st*ct * ex1

        if K == 2 and Q == 2:
            return sqrt3/4 * st**2 * ex2

    # ------------------
    # STOKES Q
    # ------------------

    if i == 1:

        if K == 2 and Q == 0:
            return -(3/(2*sqrt2))*st**2

        if K == 2 and Q == 1:
            return -(sqrt3/2) * (c2*ct + 1j*s2) * st * ex1

        if K == 2 and Q == 2:
            return -(sqrt3/4) * (
                c2*(1+ct**2)
                + 2j*s2*ct
            ) * ex2

    # ------------------
    # STOKES U
    # ------------------

    if i == 2:

        if K == 2 and Q == 0:
            return (3/(2*sqrt2))*s2*st**2

        if K == 2 and Q == 1:
            return (sqrt3/2) * (
                s2*ct - 1j*c2
            ) * st * ex1

        if K == 2 and Q == 2:
            return (sqrt3/4) * (
                s2*(1+ct**2)
                - 2j*c2*ct
            ) * ex2

    # ------------------
    # STOKES V
    # ------------------

    if i == 3:

        if K == 1 and Q == 0:
            return sqrt3/2 * ct

        if K == 1 and Q == 1:
            return -sqrt3/2 * st * ex1

    return 0.0 + 0.0j

def hanle_polarization_passive(
        Hu,
        theta_B,
        chi_B,
        theta_obs,
        chi_obs,
        w=1.0):
    """
    Compute emergent Q/I and U/I for a two-level atom
    in the Hanle effect regime.

    Parameters
    ----------
    Hu : float
        Reduced magnetic field.

    theta_B, chi_B : float
        Magnetic field inclination and azimuth (radians).

    theta_obs, chi_obs : float
        LOS inclination and azimuth (radians).

    w : float
        Radiation anisotropy factor:
            w = sqrt(2) J^2_0 / J^0_0

    Returns
    -------
    pQ, pU : floats
        Fractional linear polarization.
    """

    def idx(Q):
        return Q + 2

    # ----------------------------------
    # Radiation field tensor
    # ----------------------------------

    rho00 = 1.0

    Jvert = np.zeros(5, dtype=complex)

    # only J^2_0 present in vertical frame
    Jvert[idx(0)] = w / np.sqrt(2)

    # ----------------------------------
    # Rotate into magnetic frame
    # ----------------------------------

    D = wigner_D2(0.0, -theta_B, -chi_B)

    Jmag = D @ Jvert

    # ----------------------------------
    # Hanle effect
    # ----------------------------------

    Qs = np.array([-2, -1, 0, 1, 2])

    rho_mag = np.array([
        Jmag[i] / (1.0 + 1j*Qs[i]*Hu)
        for i in range(5)
    ])

    # ----------------------------------
    # Rotate back
    # ----------------------------------

    rho_vert = D.conj().T @ rho_mag

    # ----------------------------------
    # Emissivities
    # ----------------------------------

    epsI = (
        T(0,0,0,theta_obs,chi_obs)
        * rho00
    )

    epsQ = 0.0j
    epsU = 0.0j

    for Q in [-2,-1,0,1,2]:

        rho = rho_vert[idx(Q)]

        epsI += (
            T(0,2,Q,theta_obs,chi_obs)
            * rho
        )

        epsQ += (
            T(1,2,Q,theta_obs,chi_obs)
            * rho
        )

        epsU += (
            T(2,2,Q,theta_obs,chi_obs)
            * rho
        )

    pQ = np.real(epsQ / epsI)
    pU = np.real(epsU / epsI)

    return pQ, pU

def hanle_polarization_active(
        Hu,
        theta_B,
        chi_B,
        theta_obs,
        chi_obs,
        w=1.0):
    """
    Compute emergent Q/I and U/I for a two-level atom
    in the Hanle effect regime.

    Parameters
    ----------
    Hu : float
        Reduced magnetic field.

    theta_B, chi_B : float
        Magnetic field inclination and azimuth (radians).

    theta_obs, chi_obs : float
        LOS inclination and azimuth (radians).

    w : float
        Radiation anisotropy factor:
            w = sqrt(2) J^2_0 / J^0_0

    Returns
    -------
    pQ, pU : floats
        Fractional linear polarization.
    """

    def idx(Q):
        return Q + 2

    # ----------------------------------
    # Radiation field tensor
    # ----------------------------------

    rho00 = 1.0

    Jvert = np.zeros(5, dtype=complex)

    # only J^2_0 present in vertical frame
    Jvert[idx(0)] = w / np.sqrt(2)

    # ----------------------------------
    # Rotate into magnetic frame
    # ----------------------------------

    D = wigner_D2(chi_B, theta_B, 0.0)

    Jmag = D @ Jvert

    # ----------------------------------
    # Hanle effect
    # ----------------------------------

    Qs = np.array([-2, -1, 0, 1, 2])

    rho_mag = np.array([
        Jmag[i] / (1.0 + 1j*Qs[i]*Hu)
        for i in range(5)
    ])

    # ----------------------------------
    # Rotate back
    # ----------------------------------

    rho_vert = D.conj().T @ rho_mag

    # ----------------------------------
    # Emissivities
    # ----------------------------------

    epsI = (
        T(0,0,0,theta_obs,chi_obs)
        * rho00
    )

    epsQ = 0.0j
    epsU = 0.0j

    for Q in [-2,-1,0,1,2]:

        rho = rho_vert[idx(Q)]

        epsI += (
            T(0,2,Q,theta_obs,chi_obs)
            * rho
        )

        epsQ += (
            T(1,2,Q,theta_obs,chi_obs)
            * rho
        )

        epsU += (
            T(2,2,Q,theta_obs,chi_obs)
            * rho
        )

    pQ = np.real(epsQ / epsI)
    pU = np.real(epsU / epsI)

    return pQ, pU

def hanle_polarization_active_T(
        Hu,
        theta_B,
        chi_B,
        theta_obs,
        chi_obs,
        w=1.0):
    """
    Compute emergent Q/I and U/I for a two-level atom
    in the Hanle effect regime.

    Parameters
    ----------
    Hu : float
        Reduced magnetic field.

    theta_B, chi_B : float
        Magnetic field inclination and azimuth (radians).

    theta_obs, chi_obs : float
        LOS inclination and azimuth (radians).

    w : float
        Radiation anisotropy factor:
            w = sqrt(2) J^2_0 / J^0_0

    Returns
    -------
    pQ, pU : floats
        Fractional linear polarization.
    """

    def idx(Q):
        return Q + 2

    # ----------------------------------
    # Radiation field tensor
    # ----------------------------------

    rho00 = 1.0

    Jvert = np.zeros(5, dtype=complex)

    # only J^2_0 present in vertical frame
    Jvert[idx(0)] = w / np.sqrt(2)

    # ----------------------------------
    # Rotate into magnetic frame
    # ----------------------------------

    D = wigner_D2(chi_B, theta_B, 0.0).T

    Jmag = D @ Jvert

    # ----------------------------------
    # Hanle effect
    # ----------------------------------

    Qs = np.array([-2, -1, 0, 1, 2])

    rho_mag = np.array([
        Jmag[i] / (1.0 + 1j*Qs[i]*Hu)
        for i in range(5)
    ])

    # ----------------------------------
    # Rotate back
    # ----------------------------------

    rho_vert = D.conj().T @ rho_mag

    # ----------------------------------
    # Emissivities
    # ----------------------------------

    epsI = (
        T(0,0,0,theta_obs,chi_obs)
        * rho00
    )

    epsQ = 0.0j
    epsU = 0.0j

    for Q in [-2,-1,0,1,2]:

        rho = rho_vert[idx(Q)]

        epsI += (
            T(0,2,Q,theta_obs,chi_obs)
            * rho
        )

        epsQ += (
            T(1,2,Q,theta_obs,chi_obs)
            * rho
        )

        epsU += (
            T(2,2,Q,theta_obs,chi_obs)
            * rho
        )

    pQ = np.real(epsQ / epsI)
    pU = np.real(epsU / epsI)

    return pQ, pU

# Fig 13.3 in Landi Degl'Innocenti & Landolfi
print("D = wigner_D2(0.0, -theta_B, -chi_B)")
for chi_deg in [0,45,90]:
    pQ,pU = hanle_polarization_passive(
        Hu=1e6,
        theta_B=np.pi/2,
        chi_B=np.radians(chi_deg),
        theta_obs=np.pi/2,
        chi_obs=0.0,
        w=1.0
    )
    print(f"chi_B = {chi_deg} deg -> pQ = {pQ:.3f}, pU = {pU:.3f}")
print("_________________________________")
print("D = wigner_d2(chi_B, theta_B, 0.0)")
for chi_deg in [0,45,90]:
    pQ,pU = hanle_polarization_active(
        Hu=1e6,
        theta_B=np.pi/2,
        chi_B=np.radians(chi_deg),
        theta_obs=np.pi/2,
        chi_obs=0.0,
        w=1.0
    )
    print(f"chi_B = {chi_deg} deg -> pQ = {pQ:.3f}, pU = {pU:.3f}")
print("_________________________________")
print("D = wigner_d2(chi_B, theta_B, 0.0).T")
for chi_deg in [0,45,90]:
    pQ,pU = hanle_polarization_active_T(
        Hu=1e6,
        theta_B=np.pi/2,
        chi_B=np.radians(chi_deg),
        theta_obs=np.pi/2,
        chi_obs=0.0,
        w=1.0
    )
    print(f"chi_B = {chi_deg} deg -> pQ = {pQ:.3f}, pU = {pU:.3f}")

print("_________________________________")
print("Direct computation of emissivities using T and rho")
def idx(Q):
    return Q + 2
theta_obs = np.radians(90)
chi_obs = np.radians(0)
rho = np.zeros(5,dtype=complex)

rho[0] = -1j*0.2165009384226491   # Q=-2
rho[1] =  0.001530892816919044 -1j*0.0015308928169190817   # Q=-1
rho[2] =  0.17678995321733607    # Q=0
rho[3] = -0.001530892816919044 -1j*0.0015308928169190817   # Q=+1
rho[4] =  1j*0.2165009384226491  # Q=+2

epsQ1 = 0

for Q in [-2,-1,0,1,2]:
    epsQ1 += T(1,2,Q,theta_obs,chi_obs) * rho[idx(Q)]

epsQ2 = np.dot(
    np.array([T(1,2,Q,theta_obs,chi_obs)
              for Q in [-2,-1,0,1,2]]),
    rho
)

print(epsQ1)
print(epsQ2)

print("_________________________________")
print("(-1)**Q * np.conj(T(1,2,-Q,theta_obs,chi_obs))")
epsQ_A = 0
epsQ_B = 0

for Q in [-2,-1,0,1,2]:

    epsQ_A += (
        T(1,2,Q,theta_obs,chi_obs)
        * rho[idx(Q)]
    )

    epsQ_B += (
        (-1)**Q
        * T(1,2,Q,theta_obs,chi_obs)
        * rho[idx(-Q)]
    )

print(epsQ_A)
print(epsQ_B)

print("_________________________________")
print("Forced analytic solution")
rho = np.zeros(5,dtype=complex)

rho[idx(-2)] = -0.30618621784789724
rho[idx(0)]  = 0.25
rho[idx(+2)] = -0.30618621784789724

epsQ = 0
epsU = 0

for Q in [-2,-1,0,1,2]:

    epsQ += T(1,2,Q,np.pi/2,0) * rho[idx(Q)]
    epsU += T(2,2,Q,np.pi/2,0) * rho[idx(Q)]

print("epsQ =", epsQ)
print("epsU =", epsU)

for Q in [-2,-1,0,1,2]:

    print("Q =", Q)
    print("T(1,2,Q,np.pi/2,0) =", T(1,2,Q,np.pi/2,0))
    print("rho[idx(Q)] =", rho[idx(Q)])
    print("T(1,2,Q,np.pi/2,0)*rho[idx(Q)] =", T(1,2,Q,np.pi/2,0)*rho[idx(Q)])
    print("\n")

for Q in [-2,-1,0,1,2]:

    print("Q =", Q)
    print("T(2,2,Q,np.pi/2,0) =", T(2,2,Q,np.pi/2,0))
    print("rho[idx(Q)] =", rho[idx(Q)])
    print("T(2,2,Q,np.pi/2,0)*rho[idx(Q)] =", T(2,2,Q,np.pi/2,0)*rho[idx(Q)])
    print("\n")


def emissivity_from_rho(rho, theta_obs, chi_obs):

    epsI = 1.0
    epsQ = 0.0j
    epsU = 0.0j

    for Q in [-2,-1,0,1,2]:

        epsI += (
            T(0,2,Q,theta_obs,chi_obs)
            * rho[idx(Q)]
        )

        epsQ += (
            T(1,2,Q,theta_obs,chi_obs)
            * rho[idx(Q)]
        )

        epsU += (
            T(2,2,Q,theta_obs,chi_obs)
            * rho[idx(Q)]
        )

    return epsI, epsQ, epsU

theta_obs = np.pi/2
for chi_deg in [0,45,90]:

    epsI,epsQ,epsU = emissivity_from_rho(
        rho,
        np.pi/2,
        np.radians(chi_deg)
    )

    print()
    print("chi =",chi_deg)

    print("Q/I =", np.real(epsQ/epsI))
    print("U/I =", np.real(epsU/epsI))

rho = np.array([
    0.612372,
    0,
   -0.5,
    0,
    0.612372
], dtype=complex)

for chi_deg in [0,45,90]:
    epsI,epsQ,epsU = emissivity_from_rho(
        rho,
        np.pi/2,
        np.radians(chi_deg)
    )

    print(chi_deg,
          np.real(epsQ),
          np.real(epsU))
    
# Deriving T_Q^K 
def Tplus(Q,theta,chi):

    D = wigner_D2(chi,theta,0.0)

    return -np.sqrt(3) * D[Q+2,4]   # m'=+2


def Tminus(Q,theta,chi):

    D = wigner_D2(chi,theta,0.0)

    return -np.sqrt(3) * D[Q+2,0]   # m'=-2


def TQ_from_D(Q,theta,chi):

    return 0.5*(Tplus(Q,theta,chi)+Tminus(Q,theta,chi))


def TU_from_D(Q,theta,chi):

    return (Tplus(Q,theta,chi)-Tminus(Q,theta,chi))/(2j)

# Comparison of T_Q^K from Table 5.6 of Landi Degl'Innocenti & Landolfi with T_Q^K from wigner D-matrix
theta=np.pi/2

for chi_deg in [0,45]:

    chi=np.radians(chi_deg)

    print("\nchi =",chi_deg)

    for Q in [-2,-1,0,1,2]:

        print(
            Q,
            T(1,2,Q,theta,chi),
            TQ_from_D(Q,theta,chi)
        )

# Another test for rows and columns of wigner D-matrix
D = wigner_D2(np.radians(45), np.pi/2, 0.0).T
rho_col = D[:,2]      # column

rho_row = D[2,:]      # row

for label,rho in [("column",rho_col),
                  ("row",rho_row)]:

    print(label)

    for chi_deg in [0,45,90]:

        epsI,epsQ,epsU = emissivity_from_rho(
            rho,
            np.pi/2,
            np.radians(chi_deg)
        )

        print(
            chi_deg,
            np.real(epsQ),
            np.real(epsU)
        )

print("--------------------------------")
print("THEORY vs CODE: saturated Hanle tensor")
print("--------------------------------")

def idx(Q):
    return Q + 2

theta_B = np.pi/2
chi_B   = np.radians(45)

# use exactly the same convention as the working code
D = wigner_D2(chi_B, theta_B, 0.0).T

# -----------------------------
# theory
# -----------------------------

rho_theory = np.zeros(5,dtype=complex)

D00 = D[idx(0),idx(0)]

for Q in [-2,-1,0,1,2]:

    rho_theory[idx(Q)] = (
    np.conj(D[idx(0),idx(Q)])
    * D00
)

# -----------------------------
# code
# -----------------------------

Jvert = np.zeros(5,dtype=complex)
Jvert[idx(0)] = 1.0

Jmag = D @ Jvert

rho_mag = np.zeros(5,dtype=complex)

# saturated Hanle:
# only Q=0 survives

rho_mag[idx(0)] = Jmag[idx(0)]

rho_code = D.conj().T @ rho_mag

# -----------------------------
# compare
# -----------------------------

print("D00 =", D00)
print()

print("THEORY")
for Q in [-2,-1,0,1,2]:
    print(Q, rho_theory[idx(Q)])

print()

print("CODE")
for Q in [-2,-1,0,1,2]:
    print(Q, rho_code[idx(Q)])

print()

print("DIFFERENCE")
for Q in [-2,-1,0,1,2]:
    print(
        Q,
        rho_code[idx(Q)] - rho_theory[idx(Q)]
    )