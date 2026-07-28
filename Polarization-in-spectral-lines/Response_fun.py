import os
import sys
import numpy as np
import matplotlib.pyplot as plt

script_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(script_dir)

from functions_prt import wigner_D2
from Radiation_fun import T, idx, Jrad_to_array, radiation_tensor
from Hanle_fun import hanle_parameter_exact
from Profile_fun import (
    Phi_generalized,
    Phi_appendix,
    damping_parameter,
    A_ul,
    default_Delta_nu_D,
)
from Rotation_fun import _los_vec, _angles_from_vec, _basis_from_angles, _rotate_vert_to_mag, _rotate_qu
from Chapter_13_magnetic_branch_plots import *

# Let's define the response function that will
# compute the response of the Stokes parameters to a perturbation in the magnetic field strength B.


B_GAUSS = 5.69
GJU = 1.0

a_voigt = damping_parameter()
hu_default = hanle_parameter_exact(B_GAUSS, GJU, A_ul)
vH_default = 1.3996e6 * B_GAUSS / default_Delta_nu_D

def response_function_B(xgrid, phi, state, B_perturbation):
    """
    Compute the response function of the Stokes parameters to a perturbation in the magnetic field strength B.

    Parameters:
    - xgrid: The frequency grid (array).
    - phi: The line profile function (array).
    - state: The state of the system (dictionary containing relevant parameters).
    - B_perturbation: The perturbation in the magnetic field strength (float).

    Returns:
    - response: The response function (array).
    """
    # Compute the original Stokes parameters
    I_original, Q_original, U_original, V_original = compute_stokes_profiles(xgrid, phi, state)

    # Perturb the magnetic field strength
    state_perturbed = state.copy()
    state_perturbed['B'] += B_perturbation

    # Compute the perturbed Stokes parameters
    I_perturbed, Q_perturbed, U_perturbed, V_perturbed = compute_stokes_profiles(xgrid, phi, state_perturbed)

    # Compute the response function as the difference between perturbed and original Stokes parameters
    response_I = I_perturbed - I_original
    response_Q = Q_perturbed - Q_original
    response_U = U_perturbed - U_original
    response_V = V_perturbed - V_original

    return response_I, response_Q, response_U, response_V

def response_function_as_derivative_B(xgrid, phi, state, B_array):
    """
    Compute the response function of the Stokes parameters to a perturbation in the magnetic field strength B
    using finite differences.

    Parameters:
    - xgrid: The frequency grid (array).
    - phi: The line profile function (array).
    - state: The state of the system (dictionary containing relevant parameters).
    - B_array: An array of magnetic field strengths (array).

    Returns:
    - response_derivative: The response function as a derivative (array).
    """
    # Initialize arrays to store the Stokes parameters for each B value
    I_array = np.zeros((len(B_array), len(xgrid)))
    Q_array = np.zeros((len(B_array), len(xgrid)))
    U_array = np.zeros((len(B_array), len(xgrid)))
    V_array = np.zeros((len(B_array), len(xgrid)))

    # Compute the Stokes parameters for each B value
    for i, B in enumerate(B_array):
        state['B'] = B
        I_array[i], Q_array[i], U_array[i], V_array[i] = compute_stokes_profiles(xgrid, phi, state)

    # Compute the response function as the derivative with respect to B
    response_I_derivative = np.gradient(I_array, B_array, axis=0)
    response_Q_derivative = np.gradient(Q_array, B_array, axis=0)
    response_U_derivative = np.gradient(U_array, B_array, axis=0)
    response_V_derivative = np.gradient(V_array, B_array, axis=0)

    return response_I_derivative, response_Q_derivative, response_U_derivative, response_V_derivative

def B_finite_difference_response(xgrid, phi, state, B_array, delta_B):
    """
    Compute the response function of the Stokes parameters to a perturbation in the magnetic field strength B
    using finite differences.

    Parameters:
    - xgrid: The frequency grid (array).
    - phi: The line profile function (array).
    - state: The state of the system (dictionary containing relevant parameters).
    - B_array: An array of magnetic field strengths (array).
    - delta_B: The perturbation in the magnetic field strength (float).

    Returns:
    - response_fd: The response function as a finite difference (array).
    """
    # Initialize arrays to store the Stokes parameters for each B value
    I_array = np.zeros((len(B_array), len(xgrid)))
    Q_array = np.zeros((len(B_array), len(xgrid)))
    U_array = np.zeros((len(B_array), len(xgrid)))
    V_array = np.zeros((len(B_array), len(xgrid)))

    # Compute the Stokes parameters for each B value
    for i, B in enumerate(B_array):
        state['B'] = B
        I_array[i], Q_array[i], U_array[i], V_array[i] = compute_stokes_profiles(xgrid, phi, state)

    # Compute the response function as a finite difference with respect to B
    response_I_fd = np.zeros_like(I_array)
    response_Q_fd = np.zeros_like(Q_array)
    response_U_fd = np.zeros_like(U_array)
    response_V_fd = np.zeros_like(V_array)

    for i in range(len(B_array)):
        if i == 0:
            # Forward difference for the first point
            response_I_fd[i] = (I_array[i + 1] - I_array[i]) / delta_B
            response_Q_fd[i] = (Q_array[i + 1] - Q_array[i]) / delta_B
            response_U_fd[i] = (U_array[i + 1] - U_array[i]) / delta_B
            response_V_fd[i] = (V_array[i + 1] - V_array[i]) / delta_B
        elif i == len(B_array) - 1:
            # Backward difference for the last point
            response_I_fd[i] = (I_array[i] - I_array[i - 1]) / delta_B
            response_Q_fd[i] = (Q_array[i] - Q_array[i - 1]) / delta_B
            response_U_fd[i] = (U_array[i] - U_array[i - 1]) / delta_B
            response_V_fd[i] = (V_array[i] - V_array[i - 1]) / delta_B
        else:
            # Central difference for the interior points
            response_I_fd[i] = (I_array[i + 1] - I_array[i - 1]) / (2 * delta_B)
            response_Q_fd[i] = (Q_array[i + 1] - Q_array[i - 1]) / (2 * delta_B)
            response_U_fd[i] = (U_array[i + 1] - U_array[i - 1]) / (2 * delta_B)
            response_V_fd[i] = (V_array[i + 1] - V_array[i - 1]) / (2 * delta_B)

    return response_I_fd / I_array, response_Q_fd / Q_array, response_U_fd / U_array, response_V_fd / V_array