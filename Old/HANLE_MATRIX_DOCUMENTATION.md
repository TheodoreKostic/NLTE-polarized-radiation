# Full Hanle Matrix in Irreducible Tensor Formalism

## Overview

This implementation provides the **full 5×5 Hanle matrix** for quadrupole (K=2) density matrix components in the irreducible tensor formalism, replacing the simplified scalar depolarization approximation.

## Mathematical Framework

### Density Matrix in Irreducible Representation

The source function for polarized radiation is decomposed as:

$$S(\mu, \chi) = \sum_{K=0}^{\infty} \sum_{Q=-K}^{K} T^K_Q(\mu, \chi) \cdot S^K_Q$$

For resonance line polarization, we primarily use K=0 (scalar) and K=2 (quadrupole) tensors:

- **K=0**: Scalar component $S^0_0$ (isotropic part)
- **K=2**: Quadrupole components $S^2_Q$ where $Q \in \{-2, -1, 0, 1, 2\}$

### Statistical Equilibrium with Hanle Effect

In the **magnetic frame** (quantization axis along **B**), the density matrix evolves as:

$$\frac{d\rho^2_Q}{dt} = -\left[\Gamma_{total} + i \cdot Q \cdot \omega_L\right] \rho^2_Q$$

where:
- $\Gamma_{total} = A_{ul} + \Gamma_{col}$ (total damping rate)
- $\omega_L = g_u \mu_B B / \hbar$ (Larmor precession frequency)
- $Q$ is the magnetic quantum number

### Solution in Steady State

At statistical equilibrium (assuming LTE radiation):

$$\rho^2_Q = \frac{\rho^2_Q \text{ (initial)}}{1 + i \cdot Q \cdot \Gamma_H}$$

where $\Gamma_H = \omega_L / \Gamma_{total}$ is the **dimensionless Hanle parameter**.

## Implementation

### Two Forms of the Hanle Matrix

#### 1. Magnetic Frame (Diagonal)

```python
H_mag = hanle_matrix_magnetic_frame(
    Gamma_rad=A_ul,
    Gamma_col=Gamma_collisional,
    omega_L=omega_L
)
```

**Properties:**
- Diagonal 5×5 matrix
- Elements: $H^{mag}_{Q,Q} = \frac{1}{1 + i \cdot Q \cdot \Gamma_H}$
- Order: [Q=-2, Q=-1, Q=0, Q=1, Q=2]

**Advantages:**
- Simple form
- Physically transparent (each Q component decouples)
- Used when magnetic field is already aligned with quantization axis

#### 2. Lab Frame (General)

```python
H_lab = hanle_matrix_lab_frame(
    Gamma_rad=A_ul,
    Gamma_col=Gamma_collisional,
    omega_L=omega_L,
    theta_B=theta_B,
    chi_B=chi_B
)
```

**Properties:**
- Full 5×5 complex matrix
- Includes Wigner rotation to arbitrary field orientation
- Transformation: $H_{lab} = D^{\dagger} H_{mag} D$
  where $D$ is the Wigner d-matrix rotation

**Advantages:**
- General field orientation
- Coupling between different Q components
- Correct physics for arbitrary geometry

## Application in Code

In the iteration loop:

```python
# Stack all 5 quadrupole components
J_arr = np.array([J_vert[q] for q in [-2,-1,0,1,2]])  # shape (5, N_tau)

# Apply Hanle matrix
S_arr = (1-epsilon) * W2 * np.dot(H_lab, J_arr)

# Convert back to dictionary
S_quad = {}
for i, Q in enumerate([-2,-1,0,1,2]):
    S_quad[Q] = S_arr[i]
```

## Key Parameters

| Parameter | Symbol | Description |
|-----------|--------|-------------|
| Natural decay rate | $A_{ul}$ | Einstein A coefficient |
| Collisional broadening | $\Gamma_{col}$ | Pressure broadening |
| Larmor frequency | $\omega_L$ | $g_u \mu_B B / \hbar$ |
| Dimensionless Hanle | $\Gamma_H$ | $\omega_L / \Gamma_{total}$ |
| Magnetic field angle | $\theta_B$ | Polar angle (0 = vertical) |
| Azimuthal angle | $\chi_B$ | Azimuthal angle |

## Physical Regimes

### Weak Field Limit ($\Gamma_H \ll 1$)
$$H_{Q,Q} \approx 1 - i \cdot Q \cdot \Gamma_H$$
- Hanle rotation negligible
- Depolarization dominates

### Hanle Resonance ($\Gamma_H \sim 1$)
$$|H_{Q,Q}| = \frac{1}{\sqrt{1 + Q^2 \Gamma_H^2}}$$
- Maximum depolarization for each Q
- Magnetic field effects most significant

### Strong Field Limit ($\Gamma_H \gg 1$)
$$|H_{Q,Q}| \approx \frac{1}{|Q| \Gamma_H}$$
- Rapidly decaying depolarization
- High-frequency oscillations in magnetic frame

## Comparison: Old vs New

| Aspect | Old (Simplified) | New (Full) |
|--------|------------------|------------|
| Equation | $(1 + i\Gamma Q)^{-1}$ per component | Full 5×5 matrix |
| Q-coupling | None | Yes (for non-vertical field) |
| Field orientation | Only vertical | Arbitrary |
| Accuracy | $\sim 10\%$ | Exact (irreducible tensors) |
| Complexity | Low | Moderate |

## Dependencies

- `numpy`: Matrix operations
- `functions_prt.py`: Wigner d-matrices, rotation functions

## Example Usage

```python
# Initialize Hanle matrices
H_mag = hanle_matrix_magnetic_frame(A_ul=1e8, Gamma_col=0, omega_L=1.4e6)
H_lab = hanle_matrix_lab_frame(A_ul=1e8, Gamma_col=0, 
                               omega_L=1.4e6, 
                               theta_B=np.pi/3, chi_B=np.pi/6)

# In iteration loop
J_arr = np.array([J_vert[q] for q in [-2,-1,0,1,2]])
S_arr = W2 * (1 - epsilon) * np.dot(H_lab, J_arr)
```

## References

1. Casini, R., & Manso Sainz, R. (2016). "The Hanle effect and its diagnostics applications"
2. Stenflo, J. O. (1994). "Magnetic Field Structure of the Photosphere"
3. Landi Degl'Innocenti, E., & Landolfi, M. (2004). "Polarization in Spectral Lines"

## Notes

- The matrix acts as a **depolarization operator**: $S^2_Q = W2 (1-\epsilon) H_{Q,Q'} J^2_{Q'}$
- For vertical fields ($\theta_B = 0$), $H_{lab}$ becomes block diagonal
- Complex values represent both amplitude reduction and phase shifts
- Proper interpretation requires understanding irreducible tensor components
