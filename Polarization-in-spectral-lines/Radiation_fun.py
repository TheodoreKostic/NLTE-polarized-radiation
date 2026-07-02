import numpy as np
import sys
import os

script_dir = os.path.abspath("/home/Code/NLTE-polarized-radiation")
#script_dir = os.path.abspath("/home/teodor/Documents/Codes/NLTE-polarized-radiation")
sys.path.append(script_dir)

from functions_prt import wigner_D2, wigner_d2

# For easier handling
sqrt2 = np.sqrt(2.0)
sqrt3 = np.sqrt(3.0)

def T(i, K, Q, theta, chi, gamma):
    """Irreducible spherical tensor T^K_Q(i,Omega)
     i = (0, 1, 2, 3) corresponds to Stokes (I, Q, U, V)
    Taken from Section 5.11, Table 5.6 in Landi Degl'Innocenti & Landolfi (2004), abbreviated as LL04
    """
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

# Anisotropy calculation
def anisotropy_factor(Jrad):
    """
    Radiation anisotropy

    w = sqrt(2) J20/J00
    """

    return (
        np.sqrt(2.0)
        * np.real(Jrad[(2,0)] / Jrad[(0,0)])
    )

# Transforming J to array-like instance
def Jrad_to_array(Jrad):

    Jarr = np.zeros(5, dtype=complex)

    for Q in [-2,-1,0,1,2]:
        Jarr[idx(Q)] = Jrad[(2,Q)]

    return Jarr

from scipy.integrate import dblquad
# ---------------------------------------------------------
# Photospheric intensity seen from height hR
# ---------------------------------------------------------

# Allen (1973) limb darkening coefficients for the Sun
# for biquadratic law: I(μ)/I(1) = 1 - u1*(1-μ) - u2*(1-μ)^2
u1 = 0.95 
u2 = -0.20

# According to adopted geometry
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
    hR = true height above the limb
    """
    return np.sqrt(
        1 - (1+hR)**2 * (1-mu**2)
    )

# Compute angle-integrated J_Q^K-s
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
    """
    Compute and pack J for delta = 0
    True height = projected height
    """
    J = {}

    J[(0,0)] = compute_JKQ(0,0,hR)

    for Q in [-2,-1,0,1,2]:
        J[(2,Q)] = compute_JKQ(2,Q,hR)

    return J


delta = np.radians(30.0)
hp = 0.073      # projected height from Fig. 13.3
hR = (1 + hp)/np.cos(delta) - 1

def radiation_tensor_delta(hp, delta):
    """
    Compute and pack J for delta ≠ 0, i.e. different angle for emergent radiation
    hp = projected height
    hR = true height
    """

    hR = (1+hp)/np.cos(delta) - 1
    J0 = radiation_tensor(hR)

    Jarr = np.zeros(5, dtype=complex)

    Jarr[idx(0)] = J0[(2,0)]

    D = wigner_D2(0.0, delta, 0.0)

    Jrot = D @ Jarr

    J = {}

    J[(0,0)] = J0[(0,0)]

    for Q in [-2,-1,0,1,2]:
        J[(2,Q)] = Jrot[idx(Q)]

    return J

# Normalization factor
# CGS constants
e = 4.80320427e-10   # statcoulomb (esu)
me = 9.10938356e-28  # g
c = 2.99792458e10    # cm/s

# atomic/atmospheric inputs
f = ...           # oscillator strength
N_l = ...         # column density (cm^-2)
Delta_nu_D = ...  # Doppler width in Hz
I0 = ...          # I_nu0(0) in same intensity units

# N = (e**2/(me*c)) * f * N_l * I0 / (Delta_nu_D**2)
# or, if you already computed tau_L:
# N_alt = tau_L * I0 / (np.sqrt(np.pi) * Delta_nu_D)