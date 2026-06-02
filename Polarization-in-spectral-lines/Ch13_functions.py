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

def hanle_polarization(
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

# ============================================================
# Testing Wigner matrices - d2 matrix
# ============================================================
print("Start of Wigner d2-matrix tests")
d = wigner_d2(0.0)

print(np.round(d.real,12))
print("---------------------------------")

d = wigner_d2(np.pi)

print(np.round(d.real,12))
print("---------------------------------")

beta = 1.234

d = wigner_d2(beta)

print(
    np.max(
        np.abs(
            d @ d.T - np.eye(5)
        )
    )
)
print("---------------------------------")

beta = 1.234

d = wigner_d2(beta)

M = d @ d.T

np.set_printoptions(precision=6,suppress=True)
print(M.real)
print("---------------------------------")

beta = 1.234

d_ref = np.array(
    wigner_d_small(2,beta),
    dtype=float
)

print(d_ref)
print("---------------------------------")
print(d - d_ref)
print("End of Wigner d2-matrix tests")
print("---------------------------------")
# ============================================================
# Full Wigner d-matrix
# ============================================================
print("Start of Wigner D2-matrix tests")
theta_B = 0.0
chi_B   = 1.234   # arbitrary

Jvert = np.array([
    0,
    0,
    1.0,
    0,
    0
], dtype=complex)

D = wigner_D2(chi_B, theta_B, 0.0).T

Jmag = D @ Jvert
print("theta_B =", np.degrees(theta_B))
print("chi_B =", np.degrees(chi_B))
print("Jmag =", Jmag)
print("---------------------------------")

theta_B = np.pi/2
chi_B   = 0

D = wigner_D2(0,np.pi/2,0)

Jmag = D @ Jvert
print("theta_B =", np.degrees(theta_B))
print("chi_B =", np.degrees(chi_B))
print("Jmag =", Jmag)
print("---------------------------------")

# Hu = 0.0
print("Hu = 0.0")
Qs = np.array([-2,-1,0,1,2])
rho_mag = np.zeros(5,dtype=complex)
Hu = 0.0
for i,Q in enumerate(Qs):
    rho_mag[i] = Jmag[i] / (1 + 1j*Q*Hu)

rho_vert = D.conj().T @ rho_mag
print("rho_vert =", rho_vert)
print("----------------------------------")

# Hu = 1.0
print("Hu = 1.0")
Qs = np.array([-2,-1,0,1,2])
rho_mag = np.zeros(5,dtype=complex)
Hu = 1.0
rho_mag = np.array([
    Jmag[i] / (1 + 1j*Qs[i]*Hu)
    for i in range(5)
])

rho_vert = D.conj().T @ rho_mag
print("rho_vert =", rho_vert)
for q,val in zip(Qs,rho_vert):
    print(q,val)
print("-----------------------------------")

# Hu = 1.0, chi_B = 0, theta_B = pi/2
print("Hu = 1.0, chi_B = 0, theta_B = pi/2")
theta_B = np.pi/2
chi_B   = 0 
D = wigner_D2(chi_B, theta_B, 0.0).T
Jvert = np.array([
    0,
    0,
    1.0,
    0,
    0
], dtype=complex)   
Jmag = D @ Jvert
Qs = np.array([-2,-1,0,1,2])
rho_mag = np.array([
    Jmag[i] / (1 + 1j*Qs[i]*Hu)
    for i in range(5)
])
rho_vert = D.conj().T @ rho_mag
print("rho_vert =", rho_vert)
print("-----------------------------------")

# Testing the symmetry relation of T^K_Q
print("Testing the symmetry relation of T^K_Q")
theta = 1.1
chi   = 0.7

for i in [0,1,2]:
    for Q in [1,2]:

        lhs = T(i,2,-Q,theta,chi)

        rhs = (-1)**Q * np.conj(
            T(i,2,Q,theta,chi)
        )

        print(i,Q,lhs-rhs)

print("End of tests of the symmetry relation of T^K_Q")
print("-----------------------------------")

# Testing emissitivity in the case of a purely vertical magnetic field
print("Testing emissitivity in the case of a purely vertical magnetic field")
Hu = 0.0
theta_B = np.pi/2
chi_B   = 0.0

rho00 = 1.0
w = anisotropy_w2(0.1)
rho20 = w/np.sqrt(2)

theta = np.pi/2
chi = 0

epsI = (
    T(0,0,0,theta,chi)*rho00
    +
    T(0,2,0,theta,chi)*rho20
)

epsQ = T(1,2,0,theta,chi)*rho20

pQ_tensor = np.real(epsQ/epsI)

pQ_book = 3*w/(4-w)

print("tensor =", pQ_tensor)
print("book   =", pQ_book)
print("difference =", pQ_tensor-pQ_book)

print(T(1,2,0,np.pi/2,0))
print(T(0,2,0,np.pi/2,0))

w = 0.3

rho00 = 1.0
rho20 = w/np.sqrt(2)

theta = np.pi/2
chi = 0

epsI = (
    T(0,0,0,theta,chi)*rho00
    +
    T(0,2,0,theta,chi)*rho20
)

epsQ = (
    T(1,2,0,theta,chi)*rho20
)

print("epsI =", epsI)
print("epsQ =", epsQ)
print("Q/I =", epsQ/epsI)
print("expected =", -3*w/(4-w))

pQ, pU = hanle_polarization(
    Hu=0.0,
    theta_B=np.pi/4,
    chi_B=0.0,
    theta_obs=np.pi/2,
    chi_obs=0.0,
    w=0.3
)

print("pQ =", pQ, "pU =", pU)

def hanle_polarization(
        Hu,
        theta_B,
        chi_B,
        theta_obs,
        chi_obs,
        w=1.0):

    def idx(Q):
        return Q + 2

    # ------------------------------------
    # Radiation tensor in vertical frame
    # ------------------------------------

    Jvert = np.zeros(5, dtype=complex)

    # only J^2_0 is present for axially
    # symmetric illumination

    Jvert[idx(0)] = w/np.sqrt(2)

    # ------------------------------------
    # Rotate to magnetic frame
    # ------------------------------------

    D = wigner_D2(chi_B, theta_B, 0.0).T

    Jmag = D @ Jvert
    print("Jmag =", Jmag)

    # ------------------------------------
    # Hanle effect in magnetic frame
    # ------------------------------------

    rho_mag = np.zeros(5, dtype=complex)

    for Q in [-2,-1,0,1,2]:

        rho_mag[idx(Q)] = (
            Jmag[idx(Q)]
            /
            (1.0 + 1j*Q*Hu)
        )
    print("rho_mag =", rho_mag)
    # ------------------------------------
    # Rotate back to vertical frame
    # ------------------------------------

    rho_vert = D.conj().T @ rho_mag

    print("rho_vert =", rho_vert)
    print("chi =", chi_B)
    print("D[2,:] =", D[2,:])
    print("rho_vert =", rho_vert)
    # ------------------------------------
    # Emissivities
    # Eq. (13.21)
    # ------------------------------------

    epsI = (
        T(0,0,0,theta_obs,chi_obs)
    )

    epsQ = 0.0j
    epsU = 0.0j

    for Q in [-2,-1,0,1,2]:

        rho = rho_vert[idx(Q)]

        epsI += (
            (-1)**Q
            * T(0,2,Q,theta_obs,chi_obs)
            * rho
        )

        epsQ += (
            (-1)**Q
            * T(1,2,Q,theta_obs,chi_obs)
            * rho
        )

        epsU += (
            (-1)**Q
            * T(2,2,Q,theta_obs,chi_obs)
            * rho
        )
        print(
        Q,
        T(1,2,Q,theta_obs,chi_obs),
        rho
        )
    return (
        np.real(epsQ/epsI),
        np.real(epsU/epsI)
    )

for Hu in [0, 0.1, 1, 10, 100]:
    pQ, pU = hanle_polarization(
        Hu,
        np.pi/2,
        0.0,
        np.pi/2,
        0.0,
        w=0.3
    )

    print(Hu, pQ, pU)

pQ, pU = hanle_polarization(
    Hu=1e6,
    theta_B=np.pi/2,
    chi_B=0.0,
    theta_obs=np.pi/2,
    chi_obs=0.0,
    w=1.0
)


for chi_deg in [0,30,45,60,90]:

    pQ, pU = hanle_polarization(
        Hu=1e6,
        theta_B=np.pi/2,
        chi_B=np.radians(chi_deg),
        theta_obs=np.pi/2,
        chi_obs=0.0,
        w=1.0
    )

    print(
        f"chi_B={chi_deg:3d}°  "
        f"Q/I={pQ:+.6f}  "
        f"U/I={pU:+.6f}"
    )

for chi_deg in [0,30,45,60]:

    chi = np.radians(chi_deg)

    D = wigner_D2(chi,np.pi/2,0)

    print(
        chi_deg,
        D[:,2]
    )

theta_B = np.pi/2
chi_B   = np.radians(30)

D = wigner_D2(chi_B, theta_B, 0)

rho_mag = np.zeros(5,dtype=complex)
rho_mag[2] = -0.3535533905932738

rho_vert = D.conj().T @ rho_mag

print(rho_vert)

for chi_deg in [0,30,45,60]:

    chi = np.radians(chi_deg)

    D = wigner_D2(chi,np.pi/2,0)

    rho_mag = np.zeros(5,dtype=complex)
    rho_mag[2] = 1.0

    rho_vert = D.conj().T @ rho_mag

    print(chi_deg, rho_vert)


def hanle_polarization_eq1321(
        Hu,
        theta_B,
        chi_B,
        theta_obs,
        chi_obs,
        w=1.0):

    D = wigner_D2(chi_B, theta_B, 0.0).T
    Dinv = D.conj().T

    J20 = w/np.sqrt(2)
    Qs = [-2,-1,0,1,2]

    def idx(Q):
        return Q + 2

    epsI = T(0,0,0,theta_obs,chi_obs)
    epsQ = 0.0j
    epsU = 0.0j

    J20 = w/np.sqrt(2)

    for Q in Qs:

        rhoQ = 0.0j
        
        for Qpp in Qs:

            rhoQ += (
                D[idx(Q),idx(Qpp)]
                *
                Dinv[idx(Qpp),idx(0)]
                *
                1.0/(1.0 + 1j*Qpp*Hu)
            )

        rhoQ *= J20
        print(Q, rhoQ)
        epsI += T(0,2,Q,theta_obs,chi_obs) * rhoQ

        epsQ += T(1,2,Q,theta_obs,chi_obs) * rhoQ

        epsU += T(2,2,Q,theta_obs,chi_obs) * rhoQ

    return (
        np.real(epsQ/epsI),
        np.real(epsU/epsI)
    )

for Hu in [0,0.1,1,10,100]:

    pQ1,pU1 = hanle_polarization(
        Hu,
        np.pi/2,
        np.radians(45),
        np.pi/2,
        0.0,
        w=1.0
    )

    pQ2,pU2 = hanle_polarization_eq1321(
        Hu,
        np.pi/2,
        np.radians(45),
        np.pi/2,
        0.0,
        w=1.0
    )

    print(Hu)
    print("old :", pQ1, pU1)
    print("eq21:", pQ2, pU2)
    print()

pee, qee = hanle_polarization_eq1321(
    Hu=100,
    theta_B=np.pi/2,
    chi_B=np.radians(45),
    theta_obs=np.pi/2,
    chi_obs=0.0,
    w=1.0
)

for Q in [-2,-1,0,1,2]:
    print("Q =",Q,
          "TU =",T(2,2,Q,np.pi/2,0),
          "TQ =",T(1,2,Q,np.pi/2,0))
    

def idx(Q):
    return Q + 2

rho = np.zeros(5,dtype=complex)

rho[0] = -1j*0.2165009384226491   # Q=-2
rho[1] =  0.001530892816919044 -1j*0.0015308928169190817   # Q=-1
rho[2] =  0.17678995321733607    # Q=0
rho[3] = -0.001530892816919044 -1j*0.0015308928169190817   # Q=+1
rho[4] =  1j*0.2165009384226491  # Q=+2


theta = np.pi/2
chi   = 0.0

epsQ = 0.0j
epsU = 0.0j

Qs = [-2,-1,0,1,2]

for Q in Qs:

    r = rho[Q+2]

    TQ = T(1,2,Q,theta,chi)
    TU = T(2,2,Q,theta,chi)

    print(
        f"Q={Q:+d}",
        "rho=",r,
        "TQ=",TQ,
        "TU=",TU,
        "Qterm=",TQ*r,
        "Uterm=",TU*r
    )

    epsQ += TQ*r
    epsU += TU*r

print()
print("epsQ =",epsQ)
print("epsU =",epsU)

for chi_obs_deg in [0,45,90]:
    pQ,pU = hanle_polarization_eq1321(
        Hu=1,
        theta_B=np.pi/2,
        chi_B=np.pi/4,
        theta_obs=np.pi/2,
        chi_obs=np.radians(chi_obs_deg),
        w=1
    )
    print(chi_obs_deg,pQ,pU)

for chi_deg in [0,45,90]:

    D = wigner_D2(
        np.radians(chi_deg),
        np.pi/2,
        0
    )

    print()
    print("chi =", chi_deg)
    print(np.round(D,6))

for chi_deg in [0,45,90]:

    chi = np.radians(chi_deg)

    D = wigner_D2(chi,np.pi/2,0)

    Jvert = np.zeros(5,dtype=complex)
    Jvert[2] = 1.0

    Jmag = D @ Jvert

    rho_mag = np.zeros(5,dtype=complex)
    rho_mag[2] = Jmag[2]      # saturated limit

    rho_vert = D.conj().T @ rho_mag

    print()
    print("chi =",chi_deg)
    print("Jmag =",np.round(Jmag,6))
    print("rho_vert =",np.round(rho_vert,6))

theta = np.pi/2

for chi_deg in [0,45,90]:

    chi = np.radians(chi_deg)

    print("chi =",chi_deg)

    for Q in [-2,-1,0,1,2]:
        print(
            Q,
            T(1,2,Q,theta,chi),
            T(2,2,Q,theta,chi)
        )

theta_B = np.pi/2
chi_B   = np.radians(45)

for Q in [-2,-1,0,1,2]:
    print(
        Q,
        rho_vert[idx(Q)],
        (-1)**Q * rho_vert[idx(-Q)]
    )

Hu       = 1.0
theta_B  = np.pi/2
chi_B    = np.pi/4      # 45 deg
theta_obs = np.pi/2
chi_obs   = 0.0
w = 1.0

def idx(Q):
    return Q + 2

# radiation tensor
Jvert = np.zeros(5,dtype=complex)
Jvert[idx(0)] = w/np.sqrt(2)

# rotate to magnetic frame
D = wigner_D2(chi_B, theta_B, 0.0).T

Jmag = D @ Jvert

# Hanle effect
rho_mag = np.zeros(5,dtype=complex)

for Q in [-2,-1,0,1,2]:
    rho_mag[idx(Q)] = (
        Jmag[idx(Q)]
        /
        (1.0 + 1j*Q*Hu)
    )

# rotate back
rho_vert = D.conj().T @ rho_mag

print("rho_vert:")
for Q in [-2,-1,0,1,2]:
    print(Q, rho_vert[idx(Q)])

print()
print("comparison of conventions")

for Q in [-2,-1,0,1,2]:

    a = rho_vert[idx(Q)]

    b = ((-1)**Q) * rho_vert[idx(-Q)]

    print(
        f"Q={Q:+d}",
        "rho(Q) =", a,
        "   (-1)^Q rho(-Q) =", b,
        "   difference =", a-b
    )

print()
print("Hermitian tensor test")

for Q in [-2,-1,0,1,2]:

    lhs = rho_vert[idx(Q)]

    rhs = ((-1)**Q) * np.conj(
        rho_vert[idx(-Q)]
    )

    print(
        f"Q={Q:+d}",
        "lhs =", lhs,
        "rhs =", rhs,
        "difference =", lhs-rhs
    )

theta = np.pi/2
chi   = np.radians(45)
print("Q    TQ                        TU")

for Q in [-2,-1,0,1,2]:

    print(
        Q,
        T(1,2,Q,theta,chi),
        T(2,2,Q,theta,chi)
    )

for chi_deg in [0,45,90]:
    chi = np.radians(chi_deg)

    print(
        chi_deg,
        T(2,2,0,np.pi/2,chi)
    )

for chi_deg in [0,45,90]:

    chi = np.radians(chi_deg)

    D = wigner_D2(chi, np.pi/2, 0)

    print()
    print("chi =", chi_deg)

    print("column Q'=0")
    print(np.round(D[:,2],6))

    print("row Q=0")
    print(np.round(D[2,:],6))

chi = np.radians(45)

D = wigner_D2(chi,np.pi/2,0)

e0 = np.zeros(5,dtype=complex)
e0[2] = 1.0

print("D @ e0")
print(np.round(D @ e0,6))

print()

print("D.conj().T @ e0")
print(np.round(D.conj().T @ e0,6))

for chi_deg in [0,45,90]:
    pQ,pU = hanle_polarization_eq1321(
        Hu=1e6,
        theta_B=np.pi/2,
        chi_B=np.radians(chi_deg),
        theta_obs=np.pi/2,
        chi_obs=0,
        w=1.0
    )
    print(chi_deg,pQ,pU)

chi = np.radians(45)

D = wigner_D2(chi,np.pi/2,0)

Qs = [-2,-1,0,1,2]

for i,Q in enumerate(Qs):

    lhs = D[i,2]

    rhs = np.exp(-1j*Q*chi) * wigner_d2(np.pi/2)[i,2]

    print(Q, lhs-rhs)

for i,Q in enumerate(Qs):

    lhs = D[2,i]

    rhs = np.exp(-1j*0*chi) * wigner_d2(np.pi/2)[2,i]

    print(Q, lhs-rhs)

theta_obs = np.pi/2
chi_obs   = 0
theta_B   = np.pi/2

for chi_deg in [0,45,90]:
    pQ,pU = hanle_polarization(
        Hu=100,
        theta_B=np.pi/2,
        chi_B=np.radians(chi_deg),
        theta_obs=np.pi/2,
        chi_obs=0,
        w=1
    )
    print(chi_deg,pQ,pU)

print('--------------------------------')
for chi_deg in [0,30,45,60,90]:

    pQ,pU = hanle_polarization(
        Hu=1e6,
        theta_B=np.pi/2,
        chi_B=np.radians(chi_deg),
        theta_obs=np.pi/2,
        chi_obs=0.0,
        w=1.0
    )

    print(
        chi_deg,
        pQ,
        pU
    )
print('--------------------------------')
for chi_deg in np.linspace(0,180,13):

    pQ,pU = hanle_polarization(
        Hu=1e6,
        theta_B=np.pi/2,
        chi_B=np.radians(chi_deg),
        theta_obs=np.pi/2,
        chi_obs=0.0,
        w=1.0
    )

    print(
        f"{chi_deg:5.1f}",
        f"{pQ:+.6f}",
        f"{pU:+.6e}"
    )
print('--------------------------------')
for Q in [-2,-1,0,1,2]:
    print(Q,
          T(1,2,Q,np.pi/2,0),
          T(1,2,Q,np.pi/2,np.pi/4))
print('--------------------------------')
for Q in [-2,-1,0,1,2]:
    print(Q,
          T(2,2,Q,np.pi/2,0),
          T(2,2,Q,np.pi/2,np.pi/4))
    
print('--------------------------------')
chi = np.radians(45)

D = wigner_D2(chi,np.pi/2,0)

Qs = [-2,-1,0,1,2]

for i,Q in enumerate(Qs):

    lhs = D[i,2]

    rhs = np.exp(-1j*Q*chi) * wigner_d2(np.pi/2)[i,2]

    print(Q, lhs-rhs)

print('--------------------------------')
for i,Q in enumerate(Qs):

    lhs = D[2,i]

    rhs = np.exp(-1j*0*chi) * wigner_d2(np.pi/2)[2,i]

    print(Q, lhs-rhs)