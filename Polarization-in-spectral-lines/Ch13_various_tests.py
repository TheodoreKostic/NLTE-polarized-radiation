import numpy as np
from sympy.physics.wigner import wigner_d_small
import sys
import os
import matplotlib.pyplot as plt
# Add the directory containing functions_prt.py to the Python path

script_dir = os.path.abspath("/home/Code/NLTE-polarized-radiation")
#script_dir = os.path.abspath("/home/teodor/Documents/Codes/NLTE-polarized-radiation")
sys.path.append(script_dir)

from functions_prt import wigner_d2, wigner_D2
from Ch13_short import anisotropy_w2, anisotropy

sqrt2 = np.sqrt(2.0)
sqrt3 = np.sqrt(3.0)

def T(i, K, Q, theta, chi, gamma):
    """
    Irreducible spherical tensor T^K_Q(i,Omega)
    following Landi Degl'Innocenti & Landolfi Table 5.6

    i = 0,1,2,3  -> I,Q,U,V
    """

    # negative-Q relation
    if Q < 0:
        return (-1)**Q * np.conj(
            T(i,K,-Q,theta,chi,gamma)
        )

    ct = np.cos(theta)
    st = np.sin(theta)

    c2 = np.cos(2*gamma)
    s2 = np.sin(2*gamma)

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
            return -(3/(2*sqrt2))*st**2 * c2

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
        gamma_obs,
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
        T(0,0,0,theta_obs,chi_obs,gamma_obs)
        * rho00
    )

    epsQ = 0.0j
    epsU = 0.0j

    for Q in [-2,-1,0,1,2]:

        rho = rho_vert[idx(Q)]

        epsI += (
            T(0,2,Q,theta_obs,chi_obs,gamma_obs)
            * rho
        )

        epsQ += (
            T(1,2,Q,theta_obs,chi_obs,gamma_obs)
            * rho
        )

        epsU += (
            T(2,2,Q,theta_obs,chi_obs,gamma_obs)
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
        gamma_obs,
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
        T(0,0,0,theta_obs,chi_obs,gamma_obs)
        * rho00
    )

    epsQ = 0.0j
    epsU = 0.0j

    for Q in [-2,-1,0,1,2]:

        rho = rho_vert[idx(Q)]

        epsI += (
            T(0,2,Q,theta_obs,chi_obs,gamma_obs)
            * rho
        )

        epsQ += (
            T(1,2,Q,theta_obs,chi_obs,gamma_obs)
            * rho
        )

        epsU += (
            T(2,2,Q,theta_obs,chi_obs,gamma_obs)
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
        gamma_obs,
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
        T(0,0,0,theta_obs,chi_obs,gamma_obs)
        * rho00
    )

    epsQ = 0.0j
    epsU = 0.0j

    for Q in [-2,-1,0,1,2]:

        rho = rho_vert[idx(Q)]

        epsI += (
            T(0,2,Q,theta_obs,chi_obs,gamma_obs)
            * rho
        )

        epsQ += (
            T(1,2,Q,theta_obs,chi_obs,gamma_obs)
            * rho
        )

        epsU += (
            T(2,2,Q,theta_obs,chi_obs,gamma_obs)
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
        gamma_obs=0.0,
        w=1.0
    )
    print(f"chi_B = {chi_deg} deg -> pQ = {pQ:.3f}, pU = {pU:.3f}")
print("_________________________________")
print("D = wigner_D2(chi_B, theta_B, 0.0)")
for chi_deg in [0,45,90]:
    pQ,pU = hanle_polarization_active(
        Hu=1e6,
        theta_B=np.pi/2,
        chi_B=np.radians(chi_deg),
        theta_obs=np.pi/2,
        chi_obs=0.0,
        gamma_obs=0.0,
        w=1.0
    )
    print(f"chi_B = {chi_deg} deg -> pQ = {pQ:.3f}, pU = {pU:.3f}")
print("_________________________________")
print("D = wigner_D2(chi_B, theta_B, 0.0).T")
for chi_deg in [0,45,90]:
    pQ,pU = hanle_polarization_active_T(
        Hu=1e6,
        theta_B=np.pi/2,
        chi_B=np.radians(chi_deg),
        theta_obs=np.pi/2,
        chi_obs=0.0,
        gamma_obs=0.0,
        w=1.0
    )
    print(f"chi_B = {chi_deg} deg -> pQ = {pQ:.3f}, pU = {pU:.3f}")

print("_________________________________")
print("Direct computation of emissivities using T and rho")
def idx(Q):
    return Q + 2
theta_obs = np.radians(90)
chi_obs = np.radians(0)
gamma_obs = np.radians(0)
rho = np.zeros(5,dtype=complex)

rho[0] = -1j*0.2165009384226491   # Q=-2
rho[1] =  0.001530892816919044 -1j*0.0015308928169190817   # Q=-1
rho[2] =  0.17678995321733607    # Q=0
rho[3] = -0.001530892816919044 -1j*0.0015308928169190817   # Q=+1
rho[4] =  1j*0.2165009384226491  # Q=+2

epsQ1 = 0

for Q in [-2,-1,0,1,2]:
    epsQ1 += T(1,2,Q,theta_obs,chi_obs,gamma_obs) * rho[idx(Q)]

epsQ2 = np.dot(
    np.array([T(1,2,Q,theta_obs,chi_obs,gamma_obs)
              for Q in [-2,-1,0,1,2]]),
    rho
)

print(epsQ1)
print(epsQ2)

print("_________________________________")
print("(-1)**Q * np.conj(T(1,2,-Q,theta_obs,chi_obs,gamma_obs))")
epsQ_A = 0
epsQ_B = 0

for Q in [-2,-1,0,1,2]:

    epsQ_A += (
        T(1,2,Q,theta_obs,chi_obs,gamma_obs)
        * rho[idx(Q)]
    )

    epsQ_B += (
        (-1)**Q
        * T(1,2,Q,theta_obs,chi_obs,gamma_obs)
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

    epsQ += T(1,2,Q,np.pi/2,0,0) * rho[idx(Q)]
    epsU += T(2,2,Q,np.pi/2,0,0) * rho[idx(Q)]

print("epsQ =", epsQ)
print("epsU =", epsU)

for Q in [-2,-1,0,1,2]:

    print("Q =", Q)
    print("T(1,2,Q,np.pi/2,0,0) =", T(1,2,Q,np.pi/2,0,0))
    print("rho[idx(Q)] =", rho[idx(Q)])
    print("T(1,2,Q,np.pi/2,0,0)*rho[idx(Q)] =", T(1,2,Q,np.pi/2,0,0)*rho[idx(Q)])
    print("\n")

for Q in [-2,-1,0,1,2]:

    print("Q =", Q)
    print("T(2,2,Q,np.pi/2,0,0) =", T(2,2,Q,np.pi/2,0,0))
    print("rho[idx(Q)] =", rho[idx(Q)])
    print("T(2,2,Q,np.pi/2,0,0)*rho[idx(Q)] =", T(2,2,Q,np.pi/2,0,0)*rho[idx(Q)])
    print("\n")


def emissivity_from_rho(rho, theta_obs, chi_obs, gamma_obs):

    epsI = 1.0
    epsQ = 0.0j
    epsU = 0.0j

    for Q in [-2,-1,0,1,2]:

        epsI += (
            T(0,2,Q,theta_obs,chi_obs, gamma_obs)
            * rho[idx(Q)]
        )

        epsQ += (
            T(1,2,Q,theta_obs,chi_obs, gamma_obs)
            * rho[idx(Q)]
        )

        epsU += (
            T(2,2,Q,theta_obs,chi_obs, gamma_obs)
            * rho[idx(Q)]
        )

    return epsI, epsQ, epsU

theta_obs = np.pi/2
gamma_obs = 0.0
for chi_deg in [0,45,90]:

    epsI,epsQ,epsU = emissivity_from_rho(
        rho,
        np.pi/2,
        np.radians(chi_deg),
        gamma_obs
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
        np.radians(chi_deg),
        np.radians(0)
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
gamma_obs = 0.0
for chi_deg in [0,45]:

    chi=np.radians(chi_deg)

    print("\nchi =",chi_deg)

    for Q in [-2,-1,0,1,2]:

        print(
            Q,
            T(1,2,Q,theta,chi,gamma_obs),
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
            np.radians(chi_deg),
            gamma_obs
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

rho = np.zeros(5,dtype=complex)

rho[idx(+2)] = 1.0
rho[idx(-2)] = 1.0

theta_obs = np.pi/2
gamma_obs = 0.0
for chi_deg in [0, 22.5, 45, 67.5, 90]:

    epsI,epsQ,epsU = emissivity_from_rho(
        rho,
        np.pi/2,
        np.radians(chi_deg),
        gamma_obs
    )

    print()
    print("chi =",chi_deg)

    print("Q/I =", np.real(epsQ/epsI))
    print("U/I =", np.real(epsU/epsI))

for chi_deg in [0,22.5,45,67.5,90]:

    epsI,epsQ,epsU = emissivity_from_rho(
        rho,
        np.pi/2,
        np.radians(chi_deg),
        gamma_obs
    )

    print()
    print("chi =",chi_deg)

    print("epsI =",epsI)
    print("epsQ =",epsQ)
    print("epsU =",epsU)

    print(
        "sqrt(Q²+U²)=",
        np.sqrt(
            np.real(epsQ)**2 +
            np.real(epsU)**2
        )
    )

# After adding gamma to T
print("After adding gamma to T")
theta_obs = np.pi/2
chi_obs = np.radians(0)
for gamma_deg in [0,22.5,45,67.5,90]:

    epsI,epsQ,epsU = emissivity_from_rho(
        rho,
        theta_obs=np.pi/2,
        chi_obs=0.0,
        gamma_obs=np.radians(gamma_deg)
    )
    print("epsI =",epsI)
    print("epsQ =",epsQ)
    print("epsU =",epsU)
    print(
        "sqrt(Q²+U²)=",
        np.sqrt(
            np.real(epsQ)**2 +
            np.real(epsU)**2
        )
    )
    print()

# On to the actual plots

chi_deg = np.linspace(0,90,361)

Hu_values = [
    0.0,
    0.1,
    1.0,
    10.0,
    1e6          # saturated
]

fig,ax = plt.subplots(1,2,figsize=(10,4))

for Hu in Hu_values:

    Q = []
    U = []

    for chi in chi_deg:

        pQ,pU = hanle_polarization_active_T(
            Hu=Hu,
            theta_B=np.pi/2,
            chi_B=np.radians(chi),
            theta_obs=np.pi/2,
            chi_obs=0.0,
            gamma_obs=0.0,
            w=1.0
        )

        Q.append(pQ)
        U.append(pU)

    label = f"Hu={Hu:g}"

    ax[0].plot(chi_deg,Q,label=label)
    ax[1].plot(chi_deg,U,label=label)

ax[0].set_xlabel(r'$\chi_B$ [deg]')
ax[0].set_ylabel(r'$Q/I$')

ax[1].set_xlabel(r'$\chi_B$ [deg]')
ax[1].set_ylabel(r'$U/I$')

ax[0].legend()

plt.tight_layout()
plt.savefig("Q_U_vs_chi_B_Hu.png", dpi=300)

chi_deg = np.linspace(0,90,361)

Q = []
U = []

for chi in chi_deg:

    pQ,pU = hanle_polarization_active_T(
        Hu=1e6,
        theta_B=np.pi/2,
        chi_B=np.radians(chi),
        theta_obs=np.pi/2,
        chi_obs=0.0,
        gamma_obs=0.0,
        w=1.0
    )

    Q.append(pQ)
    U.append(pU)

plt.figure(figsize=(6,4))

plt.plot(chi_deg,Q,label='Q/I')
plt.plot(chi_deg,U,label='U/I')

plt.xlabel(r'$\chi_B$ [deg]')
plt.ylabel('fractional polarization')

plt.legend()
plt.grid(True)
plt.savefig("Q_U_vs_chi_B.png", dpi=300)

P = np.sqrt(np.array(Q)**2 + np.array(U)**2)

plt.figure()

plt.plot(chi_deg,P)

plt.xlabel(r'$\chi_B$ [deg]')
plt.ylabel(r'$\sqrt{(Q/I)^2+(U/I)^2}$')

plt.grid(True)
plt.savefig("P_vs_chi_B.png", dpi=300)

# STILL no good
print("--------------------------------")
print("SATURATED HANLE DIAGRAM")
print("--------------------------------")

Hu = 1e6
theta_B = np.pi/2

chi_B_grid = np.linspace(0, np.pi, 361)

for gamma_deg in [0, 45, 90]:

    PU = []
    PQ = []

    for chi_B in chi_B_grid:

        pQ, pU = hanle_polarization_active_T(
            Hu=Hu,
            theta_B=theta_B,
            chi_B=chi_B,
            theta_obs=np.pi/2,
            chi_obs=0.0,
            gamma_obs=np.radians(gamma_deg),
            w=1.0
        )

        PQ.append(pQ)
        PU.append(pU)

    plt.figure(figsize=(6,6))
    plt.plot(PU, PQ)

    plt.xlabel("U/I")
    plt.ylabel("Q/I")

    plt.title(
        f"Saturated Hanle, gamma={gamma_deg} deg"
    )

    plt.axis("equal")
    plt.grid(True)

plt.savefig("Hanle_diagram_gamma.png", dpi=300)

print("--------------------------------")
print("MAX/MIN U IN SATURATED LIMIT")
print("--------------------------------")

Hu = 1e6

Uvals = []

for chi_B in np.linspace(0,np.pi,361):

    pQ,pU = hanle_polarization_active_T(
        Hu=Hu,
        theta_B=np.pi/2,
        chi_B=chi_B,
        theta_obs=np.pi/2,
        chi_obs=0.0,
        gamma_obs=0.0,
        w=1.0
    )

    Uvals.append(pU)

print("min(U/I) =", np.min(Uvals))
print("max(U/I) =", np.max(Uvals))

print("========================================")
print("FIGURE 13.3 DIAGNOSTIC")
print("========================================")

Hu_values = [
    0.08,
    0.16,
    0.25,
    0.36,
    0.50,
    0.69,
    0.98,
    1.54,
    3.16
]

chi_B_grid = np.linspace(0, np.pi, 721)

# --------------------------------------------------
# FIRST: saturated limit
# --------------------------------------------------

print()
print("========================================")
print("SATURATED LIMIT (Hu = 1e6)")
print("========================================")

for chi_deg in [0,30,60,90,120,150,180]:

    pQ,pU = hanle_polarization_active_T(
        Hu=1e6,
        theta_B=np.pi/2,
        chi_B=np.radians(chi_deg),
        theta_obs=np.pi/2,
        chi_obs=0.0,
        gamma_obs=0.0,
        w=1.0
    )

    print(
        f"chi_B={chi_deg:3d} deg : "
        f"Q/I={pQ:+.6f}   "
        f"U/I={pU:+.6f}"
    )

# --------------------------------------------------
# check sign of U
# --------------------------------------------------

Uvals = []
Qvals = []

for chi_B in chi_B_grid:

    pQ,pU = hanle_polarization_active_T(
        Hu=1e6,
        theta_B=np.pi/2,
        chi_B=chi_B,
        theta_obs=np.pi/2,
        chi_obs=0.0,
        gamma_obs=0.0,
        w=1.0
    )

    Uvals.append(pU)
    Qvals.append(pQ)

print()
print("Saturated extrema")
print(
    "Q/I :",
    np.min(Qvals),
    np.max(Qvals)
)
print(
    "U/I :",
    np.min(Uvals),
    np.max(Uvals)
)

# --------------------------------------------------
# Hanle diagram
# --------------------------------------------------

plt.figure(figsize=(8,8))

for Hu in Hu_values:

    PU = []
    PQ = []

    for chi_B in chi_B_grid:

        pQ,pU = hanle_polarization_active_T(
            Hu=Hu,
            theta_B=np.pi/2,
            chi_B=chi_B,
            theta_obs=np.pi/2,
            chi_obs=0.0,
            gamma_obs=0.0,
            w=1.0
        )

        PU.append(pU)
        PQ.append(pQ)

    PU = np.array(PU)
    PQ = np.array(PQ)

    print()
    print("--------------------------------")
    print(f"Hu = {Hu}")
    print("--------------------------------")

    print(
        "Q range:",
        np.min(PQ),
        np.max(PQ)
    )

    print(
        "U range:",
        np.min(PU),
        np.max(PU)
    )

    print(
        "max polarization:",
        np.max(
            np.sqrt(PQ**2 + PU**2)
        )
    )

    plt.plot(
        PU,
        PQ,
        label=f"{Hu}"
    )

# --------------------------------------------------
# mark chi_B positions on saturated curve
# --------------------------------------------------

Hu = 1e6

for chi_deg in [0,30,60,90,120,150,180]:

    pQ,pU = hanle_polarization_active_T(
        Hu=Hu,
        theta_B=np.pi/2,
        chi_B=np.radians(chi_deg),
        theta_obs=np.pi/2,
        chi_obs=0.0,
        gamma_obs=0.0,
        w=1.0
    )

    plt.plot(
        pU,
        pQ,
        marker='o'
    )

    plt.text(
        pU,
        pQ,
        f"{chi_deg}°"
    )

plt.xlabel(r"$U/I$")
plt.ylabel(r"$Q/I$")

plt.title(
    r"Hanle diagram ($\theta_B=90^\circ$)"
)

plt.grid(True)
plt.axis("equal")
plt.legend()
plt.savefig("Hanle_diagram.png", dpi=300)

# MMMM<>>>
print()
print("========================================")
print("TEST: D-DERIVED EMISSIVITY TENSORS")
print("========================================")

# --------------------------------------------------
# D-derived emissivity tensors
# --------------------------------------------------

def TQ_from_D(Q, theta, chi):

    D = wigner_D2(chi, theta, 0.0)

    Tplus  = -np.sqrt(3.0) * D[Q+2, 4]
    Tminus = -np.sqrt(3.0) * D[Q+2, 0]

    return 0.5 * (Tplus + Tminus)


def TU_from_D(Q, theta, chi):

    D = wigner_D2(chi, theta, 0.0)

    Tplus  = -np.sqrt(3.0) * D[Q+2, 4]
    Tminus = -np.sqrt(3.0) * D[Q+2, 0]

    return (Tplus - Tminus)/(2j)


# --------------------------------------------------
# emissivity using D-derived tensors
# --------------------------------------------------

def emissivity_from_rho_D(rho, theta_obs, chi_obs):

    def idx(Q):
        return Q + 2

    epsQ = 0.0j
    epsU = 0.0j

    for Q in [-2,-1,0,1,2]:

        epsQ += (
            TQ_from_D(
                Q,
                theta_obs,
                chi_obs
            )
            * rho[idx(Q)]
        )

        epsU += (
            TU_from_D(
                Q,
                theta_obs,
                chi_obs
            )
            * rho[idx(Q)]
        )

    return epsQ, epsU

print()
print("--------------------------------")
print("SATURATED HANLE TENSOR")
print("--------------------------------")

theta_B = np.pi/2
chi_B   = np.radians(45)

D = wigner_D2(
        chi_B,
        theta_B,
        0.0
    ).T

rho = np.zeros(5,dtype=complex)

Jvert = np.zeros(5,dtype=complex)
Jvert[2] = 1.0

Jmag = D @ Jvert

rho_mag = np.zeros(5,dtype=complex)
rho_mag[2] = Jmag[2]

rho = D.conj().T @ rho_mag

for Q in [-2,-1,0,1,2]:

    print(
        Q,
        rho[Q+2]
    )

print()
print("--------------------------------")
print("TABLE 5.6 vs D-DERIVED")
print("--------------------------------")

theta = np.pi/2

for chi_deg in [0,45]:

    chi = np.radians(chi_deg)

    print()
    print("chi =",chi_deg)

    for Q in [-2,-1,0,1,2]:

        print(
            "Q =",Q
        )

        print(
            "Table TQ =",
            T(
                1,
                2,
                Q,
                theta,
                chi,
                0.0
            )
        )

        print(
            "D-derived TQ =",
            TQ_from_D(
                Q,
                theta,
                chi
            )
        )

        print(
            "Table TU =",
            T(
                2,
                2,
                Q,
                theta,
                chi,
                0.0
            )
        )

        print(
            "D-derived TU =",
            TU_from_D(
                Q,
                theta,
                chi
            )
        )

        print()

print()
print("--------------------------------")
print("HANLE DIAGRAM USING D-DERIVED T")
print("--------------------------------")

Hu_values = [
    0.08,
    0.16,
    0.25,
    0.36,
    0.50,
    0.69,
    0.98,
    1.54,
    3.16,
    1e6
]

plt.figure(figsize=(7,7))

for Hu in Hu_values:

    Pq = []
    Pu = []

    for chi_deg in np.linspace(
            0,
            180,
            361
        ):

        chi_B = np.radians(chi_deg)

        # ----------------------------------
        # radiation tensor
        # ----------------------------------

        D = wigner_D2(
                chi_B,
                np.pi/2,
                0.0
            ).T

        Jvert = np.zeros(
                    5,
                    dtype=complex
                )

        Jvert[2] = 1.0

        Jmag = D @ Jvert

        rho_mag = np.zeros(
                    5,
                    dtype=complex
                )

        Qs = [-2,-1,0,1,2]

        for i,Q in enumerate(Qs):

            rho_mag[i] = (
                Jmag[i]
                /
                (1 + 1j*Q*Hu)
            )

        rho = (
            D.conj().T
            @ rho_mag
        )

        epsQ,epsU = (
            emissivity_from_rho_D(
                rho,
                np.pi/2,
                0.0
            )
        )

        Pq.append(
            np.real(epsQ)
        )

        Pu.append(
            np.real(epsU)
        )

    plt.plot(
        Pu,
        Pq,
        lw=1.5,
        label=f"H={Hu}"
    )

plt.xlabel("P_U")
plt.ylabel("P_Q")

plt.title(
    "Hanle diagram using D-derived tensors"
)

plt.legend(
    fontsize=8
)

plt.grid(True)

plt.axis("equal")
plt.savefig("Hanle_diagram_D_derived.png", dpi=300)

def emissivity_from_rho_alt(
    rho,
    theta_obs,
    chi_obs,
    gamma_obs
):

    epsI = 1.0
    epsQ = 0.0j
    epsU = 0.0j

    for Q in [-2,-1,0,1,2]:

        phase = (-1)**Q

        epsI += (
            phase
            * T(0,2,Q,
                theta_obs,
                chi_obs,
                gamma_obs)
            * rho[idx(-Q)]
        )

        epsQ += (
            phase
            * T(1,2,Q,
                theta_obs,
                chi_obs,
                gamma_obs)
            * rho[idx(-Q)]
        )

        epsU += (
            phase
            * T(2,2,Q,
                theta_obs,
                chi_obs,
                gamma_obs)
            * rho[idx(-Q)]
        )

    return epsI, epsQ, epsU

print()
print("--------------------------------")
print("CONTRACTION TEST")
print("--------------------------------")

for chi_deg in [0,30,60,90]:

    chi_B = np.radians(chi_deg)

    D = wigner_D2(
        chi_B,
        np.pi/2,
        0.0
    ).T

    Jvert = np.zeros(5,dtype=complex)
    Jvert[2] = 1.0

    Jmag = D @ Jvert

    rho_mag = np.zeros(5,dtype=complex)
    rho_mag[2] = Jmag[2]

    rho = D.conj().T @ rho_mag

    epsI1,epsQ1,epsU1 = emissivity_from_rho(
        rho,
        np.pi/2,
        0.0,
        np.pi/2
    )

    epsI2,epsQ2,epsU2 = emissivity_from_rho_alt(
        rho,
        np.pi/2,
        0.0,
        np.pi/2
    )

    print()
    print("chi_B =",chi_deg)

    print(
        "OLD :",
        np.real(epsQ1/epsI1),
        np.real(epsU1/epsI1)
    )

    print(
        "ALT :",
        np.real(epsQ2/epsI2),
        np.real(epsU2/epsI2)
    )

print()
print("========================================")
print("CONTRACTION TEST")
print("========================================")

# saturated rho in magnetic frame
Hu = 1e6
theta_B = np.pi/2
chi_B = 0.0

D = wigner_D2(
        chi_B,
        np.pi/2,
        0.0
    ).T

Jvert = np.zeros(5,dtype=complex)
Jvert[2] = 1.0

Jmag = D @ Jvert

print()
print("Jmag")
for Q in [-2,-1,0,1,2]:
    print(Q, Jmag[idx(Q)])

rho_mag = np.zeros(5,dtype=complex)
rho_mag[2] = Jmag[2]

rho = D.conj().T @ rho_mag

print("rho_Q")
for Q in [-2,-1,0,1,2]:
    print(Q, rho[idx(Q)])

print()

chis = np.linspace(0, np.pi, 181)

PU_direct = []
PQ_direct = []

PU_conj = []
PQ_conj = []

for chi in chis:

    # -------------------------------
    # DIRECT contraction
    # -------------------------------

    epsQ = 0j
    epsU = 0j

    for Q in [-2,-1,0,1,2]:

        epsQ += (
            T(1,2,Q,
              np.pi/2,
              chi,
              0.0)
            * rho[idx(Q)]
        )

        epsU += (
            T(2,2,Q,
              np.pi/2,
              chi,
              0.0)
            * rho[idx(Q)]
        )

    PQ_direct.append(np.real(epsQ))
    PU_direct.append(np.real(epsU))

    # -------------------------------
    # CONJUGATED contraction
    # -------------------------------

    epsQ = 0j
    epsU = 0j

    for Q in [-2,-1,0,1,2]:

        phase = (-1)**Q

        epsQ += (
            phase
            * T(1,2,Q,
                np.pi/2,
                chi,
                0.0)
            * rho[idx(-Q)]
        )

        epsU += (
            phase
            * T(2,2,Q,
                np.pi/2,
                chi,
                0.0)
            * rho[idx(-Q)]
        )

    PQ_conj.append(np.real(epsQ))
    PU_conj.append(np.real(epsU))

print()
print("DIRECT")
print("PQ range:",
      min(PQ_direct),
      max(PQ_direct))
print("PU range:",
      min(PU_direct),
      max(PU_direct))

print()
print("CONJUGATED")
print("PQ range:",
      min(PQ_conj),
      max(PQ_conj))
print("PU range:",
      min(PU_conj),
      max(PU_conj))

plt.figure(figsize=(7,7))

plt.plot(PU_direct,
         PQ_direct,
         label="direct")

plt.plot(PU_conj,
         PQ_conj,
         '--',
         label="(-1)^Q rho[-Q]")

plt.xlabel("P_U")
plt.ylabel("P_Q")
plt.legend()
plt.axis('equal')
plt.grid()

plt.savefig("Hanle_diagram_contraction_test.png", dpi=300)

print()
print("========================================")
print("ROTATION DIAGNOSTIC")
print("========================================")

theta_B = np.pi/2

for chi_deg in [0, 30, 60, 90]:

    chi_B = np.deg2rad(chi_deg)

    print()
    print("========================================")
    print(f"chi_B = {chi_deg} deg")
    print("========================================")

    # ----------------------------------
    # D matrix
    # ----------------------------------

    D = wigner_D2(
            chi_B,
            theta_B,
            0.0
        ).T

    # ----------------------------------
    # vertical radiation tensor
    # J^2_0 = 1
    # ----------------------------------

    Jvert = np.zeros(5, dtype=complex)
    Jvert[idx(0)] = 1.0

    print()
    print("Jvert")
    for Q in [-2,-1,0,1,2]:
        print(Q, Jvert[idx(Q)])

    # ----------------------------------
    # rotate to magnetic frame
    # ----------------------------------

    Jmag = D @ Jvert

    print()
    print("Jmag = D @ Jvert")

    for Q in [-2,-1,0,1,2]:
        print(Q, Jmag[idx(Q)])

    # ----------------------------------
    # saturated Hanle:
    # keep only Q=0
    # ----------------------------------

    rho_mag = np.zeros(5, dtype=complex)
    rho_mag[idx(0)] = Jmag[idx(0)]

    print()
    print("rho_mag after projection")

    for Q in [-2,-1,0,1,2]:
        print(Q, rho_mag[idx(Q)])

    # ----------------------------------
    # rotate back
    # ----------------------------------

    rho = D.conj().T @ rho_mag

    print()
    print("rho = D† @ rho_mag")

    for Q in [-2,-1,0,1,2]:
        print(Q, rho[idx(Q)])

    # ----------------------------------
    # check conjugation relation
    # ----------------------------------

    print()
    print("Conjugation check")

    for Q in [1,2]:

        lhs = rho[idx(-Q)]

        rhs = ((-1)**Q) * np.conj(rho[idx(Q)])

        print(
            f"Q={Q}",
            "lhs =", lhs,
            "rhs =", rhs,
            "diff =", lhs-rhs
        )

    # ----------------------------------
    # predicted phase behaviour
    # ----------------------------------

    print()
    print("rho(+2) amplitude/phase")

    amp = np.abs(rho[idx(2)])
    phase = np.angle(rho[idx(2)], deg=True)

    print("amp   =", amp)
    print("phase =", phase)

print("========================================")
print("PURE ±2 COHERENCE TEST")
print("========================================")

rho = np.zeros(5,dtype=complex)

rho[idx(+2)] = 1.0
rho[idx(-2)] = 1.0

theta_obs = np.pi/2
gamma_obs = 0.0

for chi_deg in [0,22.5,45,67.5,90]:

    chi = np.radians(chi_deg)

    epsQ = 0j
    epsU = 0j

    print()
    print("chi =",chi_deg)

    for Q in [-2,-1,0,1,2]:

        tq = T(
            1,2,Q,
            theta_obs,
            chi,
            gamma_obs
        )

        tu = T(
            2,2,Q,
            theta_obs,
            chi,
            gamma_obs
        )

        contribQ = tq * rho[idx(Q)]
        contribU = tu * rho[idx(Q)]

        epsQ += contribQ
        epsU += contribU

        print(
            "Q=",Q,
            " TQ=",tq,
            " TU=",tu
        )

    print()

    print("epsQ =",epsQ)
    print("epsU =",epsU)

    angle = np.degrees(
        0.5*np.arctan2(
            np.real(epsU),
            np.real(epsQ)
        )
    )

    amp = np.sqrt(
        np.real(epsQ)**2 +
        np.real(epsU)**2
    )

    print("amp   =",amp)
    print("angle =",angle)

print("========================================")
print("PURE ±2 COHERENCE TEST from D-derived T")
print("========================================")

rho = np.zeros(5,dtype=complex)

rho[idx(+2)] = 1.0
rho[idx(-2)] = 1.0

theta_obs = np.pi/2
gamma_obs = 0.0

for chi_deg in [0,22.5,45,67.5,90]:

    chi = np.radians(chi_deg)

    epsQ = 0j
    epsU = 0j

    print()
    print("chi =",chi_deg)

    for Q in [-2,-1,0,1,2]:

        tq = TQ_from_D(
            Q,
            theta_obs,
            chi,
        )

        tu = TU_from_D(
            Q,
            theta_obs,
            chi
        )

        contribQ = tq * rho[idx(Q)]
        contribU = tu * rho[idx(Q)]

        epsQ += contribQ
        epsU += contribU

        print(
            "Q=",Q,
            " TQ=",tq,
            " TU=",tu
        )

    print()

    print("epsQ =",epsQ)
    print("epsU =",epsU)

    angle = np.degrees(
        0.5*np.arctan2(
            np.real(epsU),
            np.real(epsQ)
        )
    )

    amp = np.sqrt(
        np.real(epsQ)**2 +
        np.real(epsU)**2
    )

    print("amp   =",amp)
    print("angle =",angle)

print("========================================")
print("HANLE PHASE DIAGNOSTIC")
print("========================================")

def idx(Q):
    return Q + 2

Hu = 0.5
theta_B = np.pi/2

for chi_deg in [0,45,90]:

    chi_B = np.radians(chi_deg)

    print()
    print("========================================")
    print(f"chi_B = {chi_deg} deg")
    print("========================================")

    # ----------------------------------
    # J^2_0 in vertical frame
    # ----------------------------------

    Jvert = np.zeros(5,dtype=complex)
    Jvert[idx(0)] = 1.0

    print()
    print("Jvert")
    for Q in [-2,-1,0,1,2]:
        print(Q, Jvert[idx(Q)])

    # ----------------------------------
    # Rotate to magnetic frame
    # ----------------------------------

    D = wigner_D2(
            chi_B,
            theta_B,
            0.0
        ).T

    Jmag = D @ Jvert

    print()
    print("Jmag")
    for Q in [-2,-1,0,1,2]:

        val = Jmag[idx(Q)]

        print(
            Q,
            val,
            " amp=",
            abs(val),
            " phase=",
            np.degrees(np.angle(val))
        )

    # ----------------------------------
    # Apply Hanle denominator
    # ----------------------------------

    Qs = np.array([-2,-1,0,1,2])

    rho_mag = np.array([
        Jmag[i] /
        (1.0 + 1j*Qs[i]*Hu)
        for i in range(5)
    ])

    print()
    print("rho_mag AFTER Hanle")

    for Q in [-2,-1,0,1,2]:

        val = rho_mag[idx(Q)]

        print(
            Q,
            val,
            " amp=",
            abs(val),
            " phase=",
            np.degrees(np.angle(val))
        )

    # ----------------------------------
    # Compare Jmag and rho_mag phases
    # ----------------------------------

    print()
    print("PHASE SHIFTS")

    for Q in [-2,-1,1,2]:

        phase_J = np.degrees(
            np.angle(
                Jmag[idx(Q)]
            )
        )

        phase_rho = np.degrees(
            np.angle(
                rho_mag[idx(Q)]
            )
        )

        print(
            f"Q={Q:2d}",
            " J phase =",
            phase_J,
            " rho phase =",
            phase_rho,
            " shift =",
            phase_rho - phase_J
        )

    # ----------------------------------
    # Rotate back to vertical frame
    # ----------------------------------

    rho_vert = D.conj().T @ rho_mag

    print()
    print("rho_vert")

    for Q in [-2,-1,0,1,2]:

        val = rho_vert[idx(Q)]

        print(
            Q,
            val,
            " amp=",
            abs(val),
            " phase=",
            np.degrees(np.angle(val))
        )

for chi_deg in [0,45,90]:

    chi = np.radians(chi_deg)

    D = wigner_D2(
            chi,
            np.pi/2,
            0.0
        )

    Jvert = np.zeros(5,dtype=complex)
    Jvert[idx(0)] = 1.0

    Jmag1 = D @ Jvert
    Jmag2 = D.T @ Jvert

    print()
    print("chi =",chi_deg)

    print("D @ Jvert")
    for Q in [-2,-1,0,1,2]:
        print(Q,Jmag1[idx(Q)])

    print()

    print("D.T @ Jvert")
    for Q in [-2,-1,0,1,2]:
        print(Q,Jmag2[idx(Q)])

plt.figure(figsize=(8,8))

for Hu in Hu_values:

    PU = []
    PQ = []

    for chi_B in chi_B_grid:

        pQ,pU = hanle_polarization_active(
            Hu=Hu,
            theta_B=np.pi/2,
            chi_B=chi_B,
            theta_obs=np.pi/2,
            chi_obs=0.0,
            gamma_obs=0.0,
            w=1.0
        )

        PU.append(pU)
        PQ.append(pQ)

    PU = np.array(PU)
    PQ = np.array(PQ)

    print()
    print("--------------------------------")
    print(f"Hu = {Hu}")
    print("--------------------------------")

    print(
        "Q range:",
        np.min(PQ),
        np.max(PQ)
    )

    print(
        "U range:",
        np.min(PU),
        np.max(PU)
    )

    print(
        "max polarization:",
        np.max(
            np.sqrt(PQ**2 + PU**2)
        )
    )

    plt.plot(
        PU,
        PQ,
        label=f"{Hu}"
    )

# --------------------------------------------------
# mark chi_B positions on saturated curve
# --------------------------------------------------

Hu = 1e6

for chi_deg in [0,30,60,90,120,150,180]:

    pQ,pU = hanle_polarization_active(
        Hu=Hu,
        theta_B=np.pi/2,
        chi_B=np.radians(chi_deg),
        theta_obs=np.pi/2,
        chi_obs=0.0,
        gamma_obs=0.0,
        w=1.0
    )

    plt.plot(
        pU,
        pQ,
        marker='o'
    )

    plt.text(
        pU,
        pQ,
        f"{chi_deg}°"
    )

plt.xlabel(r"$U/I$")
plt.ylabel(r"$Q/I$")

plt.title(
    r"Hanle diagram ($\theta_B=90^\circ$)"
)

plt.grid(True)
plt.axis("equal")
plt.legend()
plt.savefig("Hanle_diagram_no_transpose.png", dpi=300)

for chi_deg in [0,45,90]:

    chi = np.radians(chi_deg)

    D = wigner_D2(chi, np.pi/2, 0.0)

    Jmag = D @ Jvert

    print()
    print("chi =", chi_deg)

    for Q in [-2,-1,0,1,2]:
        z = Jmag[idx(Q)]
        print(
            Q,
            abs(z),
            np.degrees(np.angle(z))
        )

print()
print("========================================")
print("CHECK Jmag AZIMUTH DEPENDENCE")
print("========================================")

def idx(Q):
    return Q + 2

Jvert = np.zeros(5,dtype=complex)
Jvert[idx(0)] = 1.0

for chi_deg in [0,45,90]:

    chi_B = np.radians(chi_deg)

    D = wigner_D2(
        chi_B,
        np.pi/2,
        0.0
    )

    Jmag = D @ Jvert

    print()
    print("chi =", chi_deg)

    for Q in [-2,-1,0,1,2]:

        z = Jmag[idx(Q)]

        print(
            Q,
            "amp =", np.abs(z),
            "phase =",
            np.degrees(np.angle(z))
        )

print()
print("========================================")
print("CHECK rho_vert DEPENDENCE")
print("========================================")

Hu = 1.0

for chi_deg in [0,45,90]:

    chi_B = np.radians(chi_deg)

    D = wigner_D2(
        chi_B,
        np.pi/2,
        0.0
    )

    Jvert = np.zeros(5,dtype=complex)
    Jvert[idx(0)] = 1.0

    Jmag = D @ Jvert

    Qs = np.array([-2,-1,0,1,2])

    rho_mag = np.array([
        Jmag[i]/(1.0 + 1j*Qs[i]*Hu)
        for i in range(5)
    ])

    rho_vert = D.conj().T @ rho_mag

    print()
    print("chi_B =", chi_deg)

    for Q in [-2,-1,0,1,2]:

        z = rho_vert[idx(Q)]

        print(
            Q,
            z,
            "amp =", np.abs(z),
            "phase =", np.degrees(np.angle(z))
        )
print()

print("========================================")
print("CHECK D FACTORIZATION")
print("========================================")

for chi_deg in [0,45,90]:

    chi = np.radians(chi_deg)

    D = wigner_D2(
        chi,
        np.pi/2,
        0.0
    )

    print()
    print("chi =", chi_deg)

    for Q in [-2,-1,0,1,2]:

        row_phase = np.exp(-1j*Q*chi)

        print("row Q =", Q)

        for Qp in [-2,-1,0,1,2]:

            lhs = D[idx(Q), idx(Qp)]

            rhs = (
                row_phase
                * D[idx(Q), idx(Qp)]
                / row_phase
            )

            print(
                Qp,
                lhs
            )


def hanle_polarization_active_rhovmag(
        Hu,
        theta_B,
        chi_B,
        theta_obs,
        chi_obs,
        gamma_obs,
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

    rho_vert = rho_mag.copy()

    # ----------------------------------
    # Emissivities
    # ----------------------------------

    epsI = (
        T(0,0,0,theta_obs,chi_obs,gamma_obs)
        * rho00
    )

    epsQ = 0.0j
    epsU = 0.0j

    for Q in [-2,-1,0,1,2]:

        rho = rho_vert[idx(Q)]

        epsI += (
            T(0,2,Q,theta_obs,chi_obs,gamma_obs)
            * rho
        )

        epsQ += (
            T(1,2,Q,theta_obs,chi_obs,gamma_obs)
            * rho
        )

        epsU += (
            T(2,2,Q,theta_obs,chi_obs,gamma_obs)
            * rho
        )

    pQ = np.real(epsQ / epsI)
    pU = np.real(epsU / epsI)

    return pQ, pU


plt.figure(figsize=(8,8))

for Hu in Hu_values:

    PU = []
    PQ = []

    for chi_B in chi_B_grid:

        pQ,pU = hanle_polarization_active_rhovmag(
            Hu=Hu,
            theta_B=np.pi/2,
            chi_B=chi_B,
            theta_obs=np.pi/2,
            chi_obs=0.0,
            gamma_obs=0.0,
            w=1.0
        )

        PU.append(pU)
        PQ.append(pQ)

    PU = np.array(PU)
    PQ = np.array(PQ)

    print()
    print("--------------------------------")
    print(f"Hu = {Hu}")
    print("--------------------------------")

    print(
        "Q range:",
        np.min(PQ),
        np.max(PQ)
    )

    print(
        "U range:",
        np.min(PU),
        np.max(PU)
    )

    print(
        "max polarization:",
        np.max(
            np.sqrt(PQ**2 + PU**2)
        )
    )

    plt.plot(
        PU,
        PQ,
        label=f"{Hu}"
    )

# --------------------------------------------------
# mark chi_B positions on saturated curve
# --------------------------------------------------

Hu = 1e6

for chi_deg in [0,30,60,90,120,150,180]:

    pQ,pU = hanle_polarization_active_rhovmag(
        Hu=Hu,
        theta_B=np.pi/2,
        chi_B=np.radians(chi_deg),
        theta_obs=np.pi/2,
        chi_obs=0.0,
        gamma_obs=0.0,
        w=1.0
    )

    plt.plot(
        pU,
        pQ,
        marker='o'
    )

    plt.text(
        pU,
        pQ,
        f"{chi_deg}°"
    )

plt.xlabel(r"$U/I$")
plt.ylabel(r"$Q/I$")

plt.title(
    r"Hanle diagram ($\theta_B=90^\circ$)"
)

plt.grid(True)
plt.axis("equal")
plt.legend()
plt.savefig("Hanle_diagram_rhovmag.png", dpi=300)