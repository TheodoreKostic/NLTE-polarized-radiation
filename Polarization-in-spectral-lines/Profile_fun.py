import numpy as np
import sys
import os
from scipy.special import wofz

script_dir = os.path.abspath("/home/Code/NLTE-polarized-radiation")
#script_dir = os.path.abspath("/home/teodor/Documents/Codes/NLTE-polarized-radiation")
sys.path.append(script_dir)

from functions_prt import wigner_D2, wigner_d2
from Radiation_fun import *


from sympy.physics.wigner import wigner_3j

# So we can get float values
def W3(j1,j2,j3,m1,m2,m3):
    return float(
        wigner_3j(
            j1,j2,j3,
            m1,m2,m3
        ).evalf()
    )


# Define Φ^QKK′​ as per Eq. (10.40) from LL04

# Doppler profile
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

def phi_complex(x):
    return wofz(x) / np.sqrt(np.pi)

def phi_transition_complex(x, Mu, Ml, vH):
    shift = (Mu - Ml) * vH
    return phi_complex(shift - x)



def Phi_generalized(x, K, Kp, Q, vH):
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
            '''
            # testing purposes
            if K == 2 and Kp == 2 and Q == 2:
                if abs(complex(term)) > 1e-10:
                    print(
                        f"Mu={Mu:2d}  Mup={Mup:2d}  "
                        f"coeff={term:+.8f}"
                    )
            '''
            
            if K == 2 and Kp == 2 and Q == 0:
                coeff = (
            (-1)**(1 + Ju - Mu + qp)
            * W3(Ju,Jl,1,-Mu,Ml,q)
            * W3(Ju,Jl,1,-Mup,Ml,qp)
            * W3(Ju,Ju,K,Mu,-Mup,-Q)
            * W3(1,1,Kp,q,-qp,-Q)
                )

               
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
                phi_transition_complex(x, Mu, Ml, vH)
                + np.conj(phi_transition_complex(x, Mup, Ml, vH))
            )

            Phi += term*profile

    return pref*Phi