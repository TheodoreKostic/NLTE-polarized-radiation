import numpy as np
import sys
import os

script_dir = os.path.abspath("/home/Code/NLTE-polarized-radiation")
#script_dir = os.path.abspath("/home/teodor/Documents/Codes/NLTE-polarized-radiation")
sys.path.append(script_dir)

from functions_prt import wigner_D2, wigner_d2
from Radiation_fun import *


# The version as of 14. 06. 2026., WIP on segmentation
def hanle_polarization_corrected(
        Hu,
        J_rad,
        theta_B,
        chi_B,
        theta_obs,
        chi_obs,
        gamma_obs = np.pi/2):
    """
    CORRECTED VERSION: Apply Hanle as full matrix operator in frame transformation.
    
    The key fix: Instead of step-wise rotation + Hanle that causes phase cancellation,
    apply the full operator H = D @ H_diag @ D† which properly couples rotation and
    depolarization together.
    
    Physics: Hanle depolarization in the magnetic frame, then rotate back to observer frame.
    """
    #print("\nJarr entering hanle_polarization_corrected")

    Jarr = Jrad_to_array(J_rad)
    #print()
    #print("ENTERING FUNCTION")
    #for Q in [-2,-1,0,1,2]:
    #    print(Q, Jarr[idx(Q)])
    
    rho00 = J_rad[(0,0)]
    
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
    #J_hanle = H_full @ Jvert

    Hfull = D @ H_diag @ D.conj().T
    rho20 = Hfull @ Jarr
    '''
    Jmag = D.conj().T @ Jarr

    rho_mag = H_diag @ Jmag

    rho_vert = D @ rho_mag
    '''
    # Compute emissivity
    epsI = rho00
    epsQ = 0.0j
    epsU = 0.0j

    for i, Q in enumerate(Qs):

        rho = rho20[idx(-Q)]

        phase = (-1.0)**Q

        epsI += phase * T(0,2,Q,
                        theta_obs,chi_obs,gamma_obs) * rho

        epsQ += phase * T(1,2,Q,
                        theta_obs,chi_obs,gamma_obs) * rho

        epsU += phase * T(2,2,Q,
                        theta_obs,chi_obs,gamma_obs) * rho

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

# Compartmentalize hanle_polarization_corrected() to
# operator, application
def hanle_operator(Hu, theta_B, chi_B):
    """
    Return the full Hanle operator

        H = D H_diag D†

    acting on the rank-2 tensor components.
    """

    Qs = np.array([-2, -1, 0, 1, 2])

    H_diag = np.diag(
        [1.0/(1.0 + 1j*Q*Hu) for Q in Qs]
    )

    D = wigner_D2(chi_B, theta_B, 0.0)

    return D @ H_diag @ D.conj().T

def apply_hanle(Jarr, Hu, theta_B, chi_B):
    """
    Apply the Hanle operator to a radiation tensor array.

    Parameters
    ----------
    Jarr : ndarray(5)
        [J^2_-2 ... J^2_+2]

    Returns
    -------
    rho : ndarray(5)
        Modified alignment components.
    """

    Hfull = hanle_operator(
        Hu,
        theta_B,
        chi_B
    )

    return Hfull @ Jarr

# Compcat Hanle effect now
def hanle_polarization_corrected(
        Hu,
        J_rad,
        theta_B,
        chi_B,
        theta_obs,
        chi_obs,
        gamma_obs = np.pi/2):
   
    Jarr = Jrad_to_array(J_rad)

    rho00 = J_rad[(0,0)]

    rho20 = apply_hanle(
        Jarr,
        Hu,
        theta_B,
        chi_B
    )
    # Compute emissivity
    epsI = rho00
    epsQ = 0.0j
    epsU = 0.0j
    Qs = np.array([-2, -1, 0, 1, 2])
    for i, Q in enumerate(Qs):

        rho = rho20[idx(-Q)]

        phase = (-1.0)**Q

        epsI += phase * T(0,2,Q,
                        theta_obs,chi_obs,gamma_obs) * rho

        epsQ += phase * T(1,2,Q,
                        theta_obs,chi_obs,gamma_obs) * rho

        epsU += phase * T(2,2,Q,
                        theta_obs,chi_obs,gamma_obs) * rho

    pQ = np.real(epsQ / epsI)
    pU = np.real(epsU / epsI)
    return pQ, pU