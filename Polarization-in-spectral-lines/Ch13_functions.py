import numpy as np
from sympy.physics.wigner import wigner_d_small
import sys
import os

# Add the directory containing functions_prt.py to the Python path

script_dir = os.path.abspath("/home/teodor/Documents/Codes/NLTE-polarized-radiation")

sys.path.append(script_dir)

from functions_prt import wigner_d2, wigner_D2


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
            return (sqrt3/2) * (c2*ct + 1j*s2) * st * ex1

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

D = wigner_D2(chi_B, theta_B, 0.0)

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