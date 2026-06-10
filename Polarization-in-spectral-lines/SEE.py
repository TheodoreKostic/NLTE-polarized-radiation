import numpy as np
import sys
import os
import matplotlib.pyplot as plt

script_dir = os.path.abspath("/home/Code/NLTE-polarized-radiation")
sys.path.append(script_dir)

from functions_prt import wigner_D2, wigner_d2

# Easier
sqrt2 = np.sqrt(2.0)
sqrt3 = np.sqrt(3.0)

def T(i, K, Q, theta, chi, gamma):
    """Irreducible spherical tensor T^K_Q(i,Omega)"""
    
    if Q < 0:
        return (-1.0)**Q * np.conj(
            T(i,K,-Q,theta,chi,gamma)
        )

    ct = np.cos(theta)
    st = np.sin(theta)
    c2 = np.cos(2*gamma)
    s2 = np.sin(2*gamma)
    ex1 = np.exp(1j*chi)
    ex2 = np.exp(2j*chi)

    if i == 0:
        if K == 0 and Q == 0:
            return 1.0
        if K == 2 and Q == 0:
            return (3*ct**2 - 1)/(2*sqrt2)
        if K == 2 and Q == 1:
            return -sqrt3/2 * st*ct * ex1
        if K == 2 and Q == 2:
            return sqrt3/4 * st**2 * ex2

    if i == 1:
        if K == 2 and Q == 0:
            return -(3/(2*sqrt2))*st**2 * c2
        if K == 2 and Q == 1:
            return -(sqrt3/2) * (c2*ct + 1j*s2) * st * ex1
        if K == 2 and Q == 2:
            return -(sqrt3/4) * (c2*(1+ct**2) + 2j*s2*ct) * ex2

    if i == 2:
        if K == 2 and Q == 0:
            return (3/(2*sqrt2))*s2*st**2
        if K == 2 and Q == 1:
            return (sqrt3/2) * (s2*ct - 1j*c2) * st * ex1
        if K == 2 and Q == 2:
            return (sqrt3/4) * (s2*(1+ct**2) - 2j*c2*ct) * ex2

    if i == 3:
        if K == 1 and Q == 0:
            return sqrt3/2 * ct
        if K == 1 and Q == 1:
            return -sqrt3/2 * st * ex1

    return 0.0 + 0.0j

# Help for tensors and Js
def idx(Q):
    return Q + 2

# with rho and J
def hanle_polarization_corrected(
        Hu,
        theta_B,
        chi_B,
        theta_obs,
        chi_obs,
        gamma_obs,
        w=1.0):
    """
    CORRECTED VERSION: Apply Hanle as full matrix operator in frame transformation.
    
    The key fix: Instead of step-wise rotation + Hanle that causes phase cancellation,
    apply the full operator H = D @ H_diag @ D† which properly couples rotation and
    depolarization together.
    
    Physics: Hanle depolarization in the magnetic frame, then rotate back to observer frame.
    """

    rho00 = 1.0
    
    # Start with only J^2_0 in vertical frame
    Jvert = np.zeros(5, dtype=complex)
    Jvert[idx(0)] = w / np.sqrt(2)
    
    # Build Hanle diagonal matrix
    Qs = np.array([-2, -1, 0, 1, 2])
    H_diag = np.diag([1.0 / (1.0 + 1j*Q*Hu) for Q in Qs])
    
    # Build Wigner D-matrix for rotation to magnetic frame
    D = wigner_D2(chi_B, theta_B, 0.0)
    
    # ==================================================
    # Apply full operator: D @ H_diag @ D†
    # This rotates to magnetic frame, applies Hanle there,
    # and rotates back to vertical frame as a single operation.
    # ==================================================
    
    D_conj_T = D.conj().T
    H_full = D @ H_diag @ D_conj_T
    
    # Apply to radiation field
    J_hanle = H_full @ Jvert
    
    # Compute emissivity
    epsI = T(0, 0, 0, theta_obs, chi_obs, gamma_obs) * rho00
    epsQ = 0.0j
    epsU = 0.0j

    for i, Q in enumerate(Qs):
        rho = J_hanle[i]
        
        epsI += T(0, 2, Q, theta_obs, chi_obs, gamma_obs) * rho
        epsQ += T(1, 2, Q, theta_obs, chi_obs, gamma_obs) * rho
        epsU += T(2, 2, Q, theta_obs, chi_obs, gamma_obs) * rho

    pQ = np.real(epsQ / epsI)
    pU = np.real(epsU / epsI)

    return pQ, pU

# Numerical approach
def hanle_parameter(B, gJu, Aul):
    """
    LL04 Eq. (10.28)-(10.29)

    Parameters
    ----------
    B : float
        Magnetic field [G]

    gJu : float
        Landé factor of upper level

    Aul : float
        Einstein A coefficient [s^-1]

    Returns
    -------
    Hu : float
        Dimensionless Hanle parameter
    """

    return 8.79e6 * gJu * B / Aul

# As per book formula
def hanle_parameter_exact(B, gJu, Aul):
    """
    Using Eq. (10.28).
    """

    nu_L = 1.3996e6 * B

    return (
        2*np.pi
        * nu_L
        * gJu
        / Aul
    )

def critical_hanle_field(gJu, Aul):
    """
    Field for which Hu = 1.
    """

    return Aul / (8.79e6 * gJu)


from scipy.integrate import dblquad
# ---------------------------------------------------------
# Photospheric intensity seen from height hR
# ---------------------------------------------------------

u1 = 0.95
u2 = -0.20

def I_mu_star(mu_star):

    return (
        1
        - u1*(1-mu_star)
        - u2*(1-mu_star**2)
    )


def mu_star_from_mu(mu, hR):
    """
    Eq. (13.17) geometry.

    mu      = cos(theta) at scattering point
    mu_star = cosine on solar surface
    """

    return np.sqrt(
        1 - (1+hR)**2 * (1-mu**2)
    )

# Js
def compute_JKQ(K, Q, hR):

    mu0 = np.sqrt(
        1 - 1/(1+hR)**2
    )

    def integrand(chi, mu):

        mu_star = mu_star_from_mu(mu, hR)

        I = I_mu_star(mu_star)

        theta = np.arccos(mu)

        return (
            T(0, K, Q, theta, chi, 0.0)
            * I
        )

    val_re = dblquad(
        lambda mu, chi: np.real(
            integrand(chi, mu)
        ),
        0.0,
        2*np.pi,
        lambda _: mu0,
        lambda _: 1.0
    )[0]

    val_im = dblquad(
        lambda mu, chi: np.imag(
            integrand(chi, mu)
        ),
        0.0,
        2*np.pi,
        lambda _: mu0,
        lambda _: 1.0
    )[0]

    return (
        val_re + 1j*val_im
    )/(4*np.pi)

def radiation_tensor(hR):

    J = {}

    J[(0,0)] = compute_JKQ(0,0,hR)

    for Q in [-2,-1,0,1,2]:
        J[(2,Q)] = compute_JKQ(2,Q,hR)

    return J

# 10.27 from 
def rhoKQ_hanle_two_level(
        JKQ,
        Hu,
        w2=1.0):
    """
    Eq. (10.27)

    Jl = 0
    Ju = 1

    JKQ : rank-2 radiation tensor
          array ordered Q=-2..2

    Returns:
        rhoQ(Q=-2..2)
    """

    rho = np.zeros(5, dtype=complex)

    prefactor = w2 / np.sqrt(3.0)

    for Q in [-2, -1, 0, 1, 2]:

        rho[idx(Q)] = (
            prefactor
            * (-1)**Q
            * JKQ[idx(-Q)]
            /
            (1.0 + 1j*Q*Hu)
        )

    return rho

JKQ = np.zeros(5, dtype=complex)
JKQ[idx(0)] = 1.0

rho = rhoKQ_hanle_two_level(
    JKQ,
    Hu=0.0
)

for Q in [-2,-1,0,1,2]:
    print(Q, rho[idx(Q)])

hR = 0.1

Jrad = radiation_tensor(hR)

print("J00 =", Jrad[(0,0)])

for Q in [-2,-1,0,1,2]:
    print(Q, Jrad[(2,Q)])


gJu = 1.0
Aul = 1e8
B = 10.0
Hu = hanle_parameter(B, gJu, Aul)

rho = rhoKQ_hanle_two_level(
    JKQ,
    Hu
)

for Q in [-2,-1,0,1,2]:
    print(
        Q,
        "J =", JKQ[idx(Q)],
        "rho =", rho[idx(Q)]
    )