import numpy as np
import sys
import os
from scipy.special import wofz
from scipy.integrate import quad
import matplotlib.pyplot as plt

script_dir = os.path.abspath("/home/Code/NLTE-polarized-radiation")
#script_dir = os.path.abspath("/home/teodor/Documents/Codes/NLTE-polarized-radiation")
sys.path.append(script_dir)

from functions_prt import wigner_D2, wigner_d2
from Radiation_fun import *


from sympy.physics.wigner import wigner_3j

# So we can get float values
def W3(j1, j2, j3, m1, m2, m3):
    return float(
        wigner_3j(
            j1,j2,j3,
            m1,m2,m3
        ).evalf()
    )

# Transition parameters
A_ul = 5 * 10**7 # s^-1
default_Delta_nu_D = 4 * 10**9 # s^-1

def damping_parameter(A_ul=A_ul, Delta_nu_D=default_Delta_nu_D):
    """
    Damping parameter a = Gamma / (4 pi Delta_nu_D)
    Gamma = A_ul for radiative damping
    """
    return A_ul / (4 * np.pi * Delta_nu_D)


def zeeman_shift_parameter(B, gJ=1.0, Delta_nu_D=default_Delta_nu_D):
    """
    Dimensionless Zeeman splitting in Doppler units.

    v_H = Delta_nu_Z / Delta_nu_D,
    with Delta_nu_Z = gJ * nu_L and nu_L = 1.3996e6 * B [Hz/G].
    """
    nu_L = 1.3996e6 * B
    return gJ * nu_L / Delta_nu_D

# ---------------------------------------------------------------------------------------------------------------------

# What if define phi step-by-step instead of using wofz?
# ---------------------------------------------------
# Physical constants
# ---------------------------------------------------

c = 2.99792458e8      # m/s

# ---------------------------------------------------
# Maxwellian
# ---------------------------------------------------

def maxwell(u):

    return np.exp(-u*u)/np.sqrt(np.pi)

# ---------------------------------------------------
# Atomic Lorentz profile in FREQUENCY units
# ---------------------------------------------------

def Phi_atomic(delta_nu, Gamma):

    gamma = Gamma/(4*np.pi)

    phi = gamma/(np.pi*(delta_nu**2 + gamma**2))

    psi = delta_nu/(np.pi*(delta_nu**2 + gamma**2))

    return phi + 1j*psi

# ---------------------------------------------------
# Literal Eq. (13.16)
# ---------------------------------------------------

def Phi_convolved(
        x,
        Gamma,
        Delta_nu_D):

    def real_integrand(u):

        delta_nu = Delta_nu_D*(x-u)

        return (
            maxwell(u)
            * np.real(
                Phi_atomic(
                    delta_nu,
                    Gamma
                )
            )
        )

    def imag_integrand(u):

        delta_nu = Delta_nu_D*(x-u)

        return (
            maxwell(u)
            * np.imag(
                Phi_atomic(
                    delta_nu,
                    Gamma
                )
            )
        )

    re = quad(
        real_integrand,
        -8,
        8,
        epsabs=1e-10,
        epsrel=1e-10
    )[0]

    im = quad(
        imag_integrand,
        -8,
        8,
        epsabs=1e-10,
        epsrel=1e-10
    )[0]

    return (re + 1j*im)


# ------------------------------------------------------------
# Zeeman-shifted transition profile
# ------------------------------------------------------------

def phi_transition_convolved(
        x,
        Mu,
        Ml,
        vH,
        Gamma,
        Delta_nu_D):

    shift = (Mu-Ml)*vH

    return Phi_convolved(
        x-shift,
        Gamma,
        Delta_nu_D
    )

# Define Φ^QKK′​ as per Eq. (10.40) from LL04

# Doppler profile
# Use Doppler profile, but complex, so that we can get the antisymmetric part needed for V.
def phi_doppler(x):
    """
    Normalized Doppler profile.
    x = (nu - nu0)/Delta_nu_D
    """
    return np.exp(-x**2)/np.sqrt(np.pi)

# Zeeman shifted
def phi_transition(x, Mu, Ml, vH):
    shift = (Mu - Ml)*vH
    
    return phi_doppler(shift - x)

# Complex
# That complex structure is what generates the antisymmetric 
# dispersion-like behavior needed for V. 
# With only a real Gaussian, V stays Gaussian and very small.

def profile_1042(
        x,
        Mu,
        Mup,
        Ml,
        Gamma,
        Delta_nu_D,
        nuL_g):
    """
    Eq. (10.42) of LL04.

    Returns the profile BEFORE Doppler convolution.
    """
    Delta_nu_D = default_Delta_nu_D
    # line-center frequencies
    nu1 = (Mu - Ml) * nuL_g
    nu2 = (Mup - Ml) * nuL_g

    # convert reduced frequency to Hz
    nu = x * Delta_nu_D

    num = (
        2*Gamma
        + 1j*nuL_g*(Mup-Mu)
    )

    den = (
        (Gamma - 1j*(nu1-nu))
        *
        (Gamma + 1j*(nu2-nu))
    )

    return num/(2*np.pi*den)

# Too costly!
def maxwell(u):
    return np.exp(-u*u)/np.sqrt(np.pi)


def Phi_generalized_convolved(x, K, Kp, Q, vH, a):

    x = np.asarray(x)

    Phi_conv = np.zeros_like(x, dtype=np.complex128)

    for i, xx in enumerate(x):

        def real_integrand(u):
            return (
                maxwell(u)
                * np.real(
                    Phi_generalized(xx-u, K, Kp, Q, vH, a)
                )
            )

        def imag_integrand(u):
            return (
                maxwell(u)
                * np.imag(
                    Phi_generalized(xx-u, K, Kp, Q, vH, a)
                )
            )

        re = quad(real_integrand, -8, 8,
                  epsabs=1e-10, epsrel=1e-10)[0]

        im = quad(imag_integrand, -8, 8,
                  epsabs=1e-10, epsrel=1e-10)[0]

        Phi_conv[i] = re + 1j*im

    return Phi_conv

# ---------------------------------------------------------------------------------------------------------------------

def phi_complex(x, a):
    return wofz(x + 1j*a) / np.sqrt(np.pi)


def phi_transition_complex(x, Mu, Ml, vH, a):
    shift = (Mu - Ml) * vH
    return phi_complex(x - shift, a)

# Main part, the generalized profile function Φ^QKK′​, as per Eq. (10.40) from LL04.
def Phi_generalized(x, K, Kp, Q, vH, a, return_pairs = False):
    """
    Eq. (10.40)

    Jl = 0
    Ju = 1

    x may be scalar or numpy array.
    """

    Ju = 1
    Jl = 0
    Ml = 0

    pref = np.sqrt(
        3*(2*Ju+1)*(2*K+1)*(2*Kp+1)
    )

    Phi = np.zeros_like(np.asarray(x), dtype=np.complex128)

    if return_pairs:
        pair_profiles = np.empty((3,3), dtype=object)

    # Loop through q and qp
    for Mu in (-1,0,1):
        for Mup in (-1,0,1):

            q  = Mu
            qp = Mup

            term = (
                (-1)**(1 + Ju - Mu + qp)

                * W3(
                    Ju, Jl, 1,
                    -Mu, Ml, q
                )

                * W3(
                    Ju, Jl, 1,
                    -Mup, Ml, qp
                )

                * W3(
                    Ju, Ju, K,
                    Mu, -Mup, -Q
                )

                * W3(
                    1, 1, Kp,
                    q, -qp, -Q
                )
            )
            
            if K == 2 and Kp == 1 and Q == 0:
                if Mu == 1 and Mup == -1:
                    q = Mu
                    qp = Mup
                    coeff = (
                (-1)**(1 + Ju - Mu + qp)
                * W3(Ju,Jl,1,-Mu,Ml,q)
                * W3(Ju,Jl,1,-Mup,Ml,qp)
                * W3(Ju,Ju,K,Mu,-Mup,-Q)
                * W3(1,1,Kp,q,-qp,-Q)
                    )

                    print("W1 =", W3(Ju,Jl,1,-Mu,Ml,q))
                    print("W2 =", W3(Ju,Jl,1,-Mup,Ml,qp))
                    print("W3 =", W3(Ju,Ju,K,Mu,-Mup,-Q))
                    print("W4 =", W3(1,1,Kp,q,-qp,-Q))

                    print(
                        "K = ", K,
                        "Kp = ", Kp,
                        "Q = ", Q,
                        "Mu =", Mu,
                        "Mup =", Mup,
                        "coeff =", coeff,
                        "vH = ", vH
                    )
              
            profile = 0.5 * (
                phi_transition_complex(x, Mu, Ml, vH, a)
                + np.conj(phi_transition_complex(x, Mup, Ml, vH, a))
            )
          
            Phi += term*profile
            if return_pairs:
                pair_profiles[Mu+1, Mup+1] = term * profile
            '''
            # testing purposes
            if K == 2 and Kp == 1 and Q == 0:
                #if abs(complex(term)) > 1e-10:
                print(
                        f"Mu={Mu:2d} "
                        f"Mup={Mup:2d} "
                        f"coeff={term:+.8f} "
                        f"profile={profile} "
                        f"Phi={Phi} "
                    )
            '''
            '''
            profile = profile_1042(
                x,
                Mu,
                Mup,
                Ml,
                A_ul/4*np.pi,
                default_Delta_nu_D,
                1.3996e6
            )
            Phi += term * profile
            '''
    if return_pairs:
        return pref*Phi, pair_profiles

    return pref*Phi

# ---------------------------------------------------------------------------------------------------------------------

# Profile function based on Appendix A13 form LL04
# Properties of Generalized Profiles
def phi_q(x, q, vH, a):
    """
    Appendix definition for q = -1,0,+1.
    We use q -> shift = -q * vH, because q is the Zeeman component label.
    """
    return np.real(phi_complex(x + q * vH, a))


def psi_q(x, q, vH, a):
    return np.imag(phi_complex(x + q * vH, a))


def Phi_appendix(x, K, Kp, Q, vH, a):
    x = np.asarray(x, dtype=np.complex128)

    phi_p1 = phi_q(x, 1, vH, a)
    phi_0 = phi_q(x, 0, vH, a)
    phi_m1 = phi_q(x, -1, vH, a)

    psi_p1 = psi_q(x, 1, vH, a)
    psi_0 = psi_q(x, 0, vH, a)
    psi_m1 = psi_q(x, -1, vH, a)

    if Q < 0:
        return np.conj(Phi_appendix(x, K, Kp, -Q, vH, a))

    if K == 0 and Kp == 0 and Q == 0:
        return (phi_p1 + phi_0 + phi_m1) / 3.0

    if K == 0 and Kp == 1 and Q == 0:
        return (phi_p1 - phi_m1) / np.sqrt(6.0)

    if K == 0 and Kp == 2 and Q == 0:
        return (phi_p1 - 2.0 * phi_0 + phi_m1) / (3.0 *np.sqrt(2.0))

    if K == 1 and Kp == 0 and Q == 0:
        return -(phi_p1 - phi_m1) / np.sqrt(6.0)

    if K == 1 and Kp == 1 and Q == 0:
        #return -0.25 * (phi_p1 + 1j * psi_p1 + 2.0 * phi_0 + phi_m1 - 1j * psi_m1)
        return -(phi_p1 + phi_m1) / 2.0

    if K == 1 and Kp == 2 and Q == 0:
        return -(phi_p1 - phi_m1) / (2.0 * np.sqrt(3.0))

    if K == 2 and Kp == 0 and Q == 0:
        #return (phi_p1 - 2.0 * phi_0 + phi_m1) / np.sqrt(3.0)
        return -(phi_p1 - 2.0 * phi_0 + phi_m1) / (3.0 *np.sqrt(2.0))

    if K == 2 and Kp == 1 and Q == 0:
        return (phi_p1 - phi_m1) / (2.0 * np.sqrt(3.0))

    if K == 2 and Kp == 2 and Q == 0:
        return (phi_p1 + 4.0 * phi_0 + phi_m1) / 6.0

    if K == 1 and Kp == 1 and Q == 1:
        return -0.25 * (phi_p1 + 1j * psi_p1 + 2.0 * phi_0 + phi_m1 - 1j * psi_m1)

    if K == 1 and Kp == 2 and Q == 1:
        return -0.25 * (phi_p1 + 1j * psi_p1 - 2.0j * psi_0 - phi_m1 + 1j * psi_m1)

    if K == 2 and Kp == 2 and Q == 1:
        return 0.25 * (phi_p1 + 1j * psi_p1 + 2.0 * phi_0 + phi_m1 - 1j * psi_m1)

    if K == 2 and Kp == 2 and Q == 2:
        return 0.5 * (phi_p1 + 1j * psi_p1 + phi_m1 - 1j * psi_m1)

    return np.zeros_like(x, dtype=np.complex128)

# After this we can go to response functions!