import numpy as np
import sys
import os
import matplotlib.pyplot as plt

script_dir = os.path.abspath("/home/Code/NLTE-polarized-radiation")
sys.path.append(script_dir)

from functions_prt import wigner_D2

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


def idx(Q):
    return Q + 2


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


# ============================================================
# TESTS
# ============================================================

print("=" * 70)
print("CORRECTED HANLE - FULL MATRIX OPERATOR")
print("=" * 70)

print()
print("TEST 1: Q/I and U/I vs chi_B")
print("-" * 70)

chi_deg_test = np.array([0, 30, 60, 90, 120, 150, 180])

for Hu in [0.0, 0.1, 1.0, 1e6]:
    print()
    print(f"Hu = {Hu:g}")
    
    for chi in chi_deg_test:
        pQ, pU = hanle_polarization_corrected(
            Hu=Hu,
            theta_B=np.pi/2,
            chi_B=np.radians(chi),
            theta_obs=np.pi/2,
            chi_obs=0.0,
            gamma_obs=0.0,
            w=1.0
        )
        print(f"  χB={chi:3d}° : Q/I = {pQ:+.6f}, U/I = {pU:+.6f}")

# Test 2: Saturated limit properties
print()
print()
print("TEST 2: Saturated limit (Hu = 1e6)")
print("-" * 70)

chi_B_grid = np.linspace(0, np.pi, 181)

Q_vals = []
U_vals = []

for chi_B in chi_B_grid:
    pQ, pU = hanle_polarization_corrected(
        Hu=1e6,
        theta_B=np.pi/2,
        chi_B=chi_B,
        theta_obs=np.pi/2,
        chi_obs=0.0,
        gamma_obs=0.0,
        w=1.0
    )
    Q_vals.append(pQ)
    U_vals.append(pU)

Q_vals = np.array(Q_vals)
U_vals = np.array(U_vals)

print(f"Q/I range: [{np.min(Q_vals):.6f}, {np.max(Q_vals):.6f}]")
print(f"U/I range: [{np.min(U_vals):.6f}, {np.max(U_vals):.6f}]")
print(f"Variation in Q: {np.max(Q_vals) - np.min(Q_vals):.6f}")
print(f"Variation in U: {np.max(U_vals) - np.min(U_vals):.6f}")

if np.max(Q_vals) - np.min(Q_vals) > 0.01 or np.max(U_vals) - np.min(U_vals) > 0.01:
    print("✓ GOOD: χB dependence is PRESENT")
else:
    print("✗ BAD: χB dependence is MISSING")

# Plot Hanle diagram
print()
print()
print("GENERATING PLOT")
print("-" * 70)

Hu_values = [0.08, 0.16, 0.25, 0.36, 0.50, 0.69, 0.98, 1.54, 3.16, 1e6]
chi_B_grid = np.linspace(0, np.pi, 721)

plt.figure(figsize=(9, 9))

for Hu in Hu_values:
    PU = []
    PQ = []

    for chi_B in chi_B_grid:
        pQ, pU = hanle_polarization_corrected(
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
    
    plt.plot(PU, PQ, lw=1.5, label=f"H={Hu:g}")

# Mark chi_B on saturated curve
Hu = 1e6
for chi_deg in [0, 45, 90, 135, 180]:
    pQ, pU = hanle_polarization_corrected(
        Hu=Hu,
        theta_B=np.pi/2,
        chi_B=np.radians(chi_deg),
        theta_obs=np.pi/2,
        chi_obs=0.0,
        gamma_obs=0.0,
        w=1.0
    )
    plt.plot(pU, pQ, 'ko', markersize=5)
    plt.text(pU + 0.01, pQ, f"{chi_deg}°", fontsize=9)

plt.xlabel(r"$U/I$", fontsize=12)
plt.ylabel(r"$Q/I$", fontsize=12)
plt.title(r"Hanle diagram (full matrix operator, $\theta_B=90°$)", fontsize=12)
plt.grid(True, alpha=0.3)
plt.axis("equal")
plt.legend(fontsize=9, loc='best')
plt.tight_layout()
plt.savefig("Ch13_corrected_Hanle_diagram.png", dpi=300)
print("Saved: Ch13_corrected_Hanle_diagram.png")

print()
print("=" * 70)

# Quick diagnostic: What happens if we apply Hanle WITHOUT rotating back?

print("\n" + "="*70)
print("DIAGNOSTIC: Hanle in magnetic frame WITHOUT back-rotation")
print("="*70)

# This tests Section 8 of the debugging summary
D = wigner_D2(np.radians(45), np.pi/2, 0.0)
Jvert = np.zeros(5, dtype=complex)
Jvert[idx(0)] = 1.0

Jmag = D @ Jvert

Qs = np.array([-2, -1, 0, 1, 2])
rho_mag = np.array([
    Jmag[i] / (1.0 + 1j*Qs[i]*1e6)  # Saturated
    for i in range(5)
])

# DON'T rotate back
rho_vert_nomag = rho_mag.copy()

print("\nUsing J_hanle in MAGNETIC frame (no back-rotation):")

epsQ = 0.0j
epsU = 0.0j

for Q in Qs:
    rho = rho_vert_nomag[idx(Q)]
    epsQ += T(1, 2, Q, np.pi/2, 0.0, 0.0) * rho
    epsU += T(2, 2, Q, np.pi/2, 0.0, 0.0) * rho

print(f"Q/I = {np.real(epsQ):.6f}")
print(f"U/I = {np.real(epsU):.6f}")

print("\nDIAGNOSTIC: Testing with FULL J_vert distribution")
print("-" * 70)

# Start with full distribution (from your test)
Jvert_full = np.zeros(5, dtype=complex)
Jvert_full[0] = -1j*0.2165009384226491   # Q=-2
Jvert_full[1] =  0.001530892816919044 -1j*0.0015308928169190817   # Q=-1
Jvert_full[2] =  0.17678995321733607    # Q=0
Jvert_full[3] = -0.001530892816919044 -1j*0.0015308928169190817   # Q=+1
Jvert_full[4] =  1j*0.2165009384226491  # Q=+2

# Now apply Hanle with full matrix operator
Qs = np.array([-2, -1, 0, 1, 2])
H_diag = np.diag([1.0 / (1.0 + 1j*Q*1e6) for Q in Qs])
D = wigner_D2(np.radians(45), np.pi/2, 0.0)
H_full = D @ H_diag @ D.conj().T
J_hanle_full = H_full @ Jvert_full

epsQ = epsU = 0.0j
for i, Q in enumerate(Qs):
    epsQ += T(1, 2, Q, np.pi/2, 0.0, 0.0) * J_hanle_full[i]
    epsU += T(2, 2, Q, np.pi/2, 0.0, 0.0) * J_hanle_full[i]

print(f"With full J_vert: Q/I = {np.real(epsQ):.6f}, U/I = {np.real(epsU):.6f}")

# 05. 06. 2026. 19:54
theta_B = np.pi/2
chi_B = np.pi/4
Hu = 1.0

A = D.conj().T @ H_diag @ D
B = D @ H_diag @ D.conj().T

print(np.round(A,6))
print(np.round(B,6))

e0 = np.zeros(5,dtype=complex)
e0[2] = 1

A @ e0
B @ e0

# 05. 06. 2026. 20:13
import numpy as np

def idx(Q):
    return Q + 2

print()
print("="*60)
print("TEST OF AZIMUTHAL PHASE LAW")
print("="*60)

theta = np.pi/2

# only Q'=0 component
J0 = np.zeros(5, dtype=complex)
J0[idx(0)] = 1.0

# reference at chi=0
D0 = wigner_D2(0.0, theta, 0.0)

Jref = D0 @ J0

for chi_deg in [45, 90]:

    chi = np.radians(chi_deg)

    D = wigner_D2(chi, theta, 0.0)

    J = D @ J0

    print()
    print(f"chi = {chi_deg}")

    for Q in [-2,-1,0,1,2]:

        if abs(Jref[idx(Q)]) < 1e-12:
            continue

        ratio = J[idx(Q)] / Jref[idx(Q)]

        phase = np.degrees(np.angle(ratio))

        expected_minus = -Q * chi_deg
        expected_plus  = +Q * chi_deg

        # wrap to [-180,180]
        expected_minus = ((expected_minus + 180) % 360) - 180
        expected_plus  = ((expected_plus  + 180) % 360) - 180

        print(
            f"Q={Q:+d}  "
            f"phase(ratio)={phase:8.2f}°   "
            f"-Qχ={expected_minus:8.2f}°   "
            f"+Qχ={expected_plus:8.2f}°"
        )

print()
print("="*60)
print("CHECK FACTORIZATION")
print("="*60)

theta = np.pi/2
chi   = np.radians(45)

D = wigner_D2(chi, theta, 0.0)
d = wigner_D2(0.0, theta, 0.0)

for Q in [-2,-1,0,1,2]:

    row = idx(Q)

    # pick Q'=0 column
    col = idx(0)

    lhs = D[row,col]

    rhs_minus = np.exp(-1j*Q*chi) * d[row,col]
    rhs_plus  = np.exp(+1j*Q*chi) * d[row,col]

    err_minus = abs(lhs-rhs_minus)
    err_plus  = abs(lhs-rhs_plus)

    print(
        f"Q={Q:+d}   "
        f"err(-iQχ)={err_minus:.3e}   "
        f"err(+iQχ)={err_plus:.3e}"
    )

# 05. 06. 2026. 20:17
Hu = 1.0
for chi_deg in [0,45,90]:

    chi = np.radians(chi_deg)

    D = wigner_D2(chi, np.pi/2, 0)

    H = np.diag([
        1/(1-2j*Hu),
        1/(1-1j*Hu),
        1,
        1/(1+1j*Hu),
        1/(1+2j*Hu)
    ])

    A = D.conj().T @ H @ D

    print()
    print("chi =", chi_deg)

    print(np.round(A,6))

# 05. 06. 2026. 20:25
print("\n" + "="*70)
print("WIGNER D MATRIX CONVENTION TEST")
print("="*70)

# Test 1: Forward-backward test
print("\nTest 1: Check D @ D† = I")
D = wigner_D2(np.radians(45), np.pi/2, 0.0)
product = D @ D.conj().T
is_identity = np.allclose(product, np.eye(5))
print(f"D @ D† ≈ I: {is_identity}")
print(f"Max deviation from identity: {np.max(np.abs(product - np.eye(5))):.3e}")

# Test 2: What matrix gives identity product?
print("\nTest 2: Find correct inverse")
D = wigner_D2(np.radians(45), np.pi/2, 0.0)
inv_candidates = [
    ("D†", D.conj().T),
    ("D.T", D.T),
    ("D^-1 (numpy)", np.linalg.inv(D)),
]

for name, candidate in inv_candidates:
    product = D @ candidate
    max_err = np.max(np.abs(product - np.eye(5)))
    print(f"  D @ {name} error: {max_err:.3e}")

# Test 3: Understanding the azimuthal phase structure
print("\nTest 3: Azimuthal phase law - which e^{iQχ} convention?")
print("-" * 70)

theta = np.pi/2
chi_ref = 0.0
chi_test = np.radians(45)

D_ref = wigner_D2(chi_ref, theta, 0.0)
D_test = wigner_D2(chi_test, theta, 0.0)

J0 = np.zeros(5, dtype=complex)
J0[2] = 1.0  # Only Q'=0

J_ref = D_ref @ J0
J_test = D_test @ J0

print("\nIf D @ J_vertical gives magnetic frame components:")
print("Expected phase: e^{-iQχ} (descending with +Q)")
print()

for Q in [-2, -1, 0, 1, 2]:
    if abs(J_ref[Q+2]) < 1e-12:
        continue
    
    ratio = J_test[Q+2] / J_ref[Q+2]
    phase_deg = np.degrees(np.angle(ratio))
    expected_minus = -Q * 45
    expected_plus = +Q * 45
    
    # Wrap to [-180, 180]
    expected_minus = ((expected_minus + 180) % 360) - 180
    expected_plus = ((expected_plus + 180) % 360) - 180
    
    match_minus = abs(phase_deg - expected_minus) < 1
    match_plus = abs(phase_deg - expected_plus) < 1
    
    print(f"Q={Q:+d}: phase={phase_deg:7.1f}° | -Qχ={expected_minus:7.1f}° {'✓' if match_minus else ''} | +Qχ={expected_plus:7.1f}° {'✓' if match_plus else ''}")

# Test 4: What's the correct back-rotation?
print("\nTest 4: Testing back-rotation options")
print("-" * 70)

chi_B = np.radians(45)
D = wigner_D2(chi_B, np.pi/2, 0.0)

# Start with J^2_0 in vertical
J_vert = np.zeros(5, dtype=complex)
J_vert[2] = 1.0

# Rotate to magnetic
J_mag = D @ J_vert

print(f"\nJ_vert[Q=0] = {J_vert[2]:.6f}")
print(f"\nAfter D @ J_vert (in magnetic frame):")
for Q in [-2, -1, 0, 1, 2]:
    print(f"  Q={Q:+d}: {J_mag[Q+2]:.6f}")

# Try different back-rotations
back_rots = [
    ("D†  = D.conj().T", D.conj().T),
    ("D.T (transpose)", D.T),
    ("D^-1", np.linalg.inv(D)),
]

print(f"\nBack-rotation tests:")
for name, back_op in back_rots:
    J_recovered = back_op @ J_mag
    recovery_error = np.max(np.abs(J_recovered - J_vert))
    print(f"  {name:20s}: recovery error = {recovery_error:.3e}")

print("\n" + "="*70)

# 05. 06. 2026. 20:29
print("\n" + "="*70)
print("INVESTIGATE ANISOTROPY FUNCTION")
print("="*70)

# Check what anisotropy returns
from Ch13_short import anisotropy_w2, anisotropy

w = 1.0

print(f"\nanisotropy_w2(w={w}) = ?")
print(f"anisotropy(w={w}) = ?")

# What we're currently using:
J_current = np.zeros(5, dtype=complex)
J_current[2] = w / np.sqrt(2)

print(f"\nCurrent J_vert initialization:")
for Q in [-2, -1, 0, 1, 2]:
    if abs(J_current[Q+2]) > 1e-10:
        print(f"  J^2_{Q}: {J_current[Q+2]:.6f}")

# The test case had:
J_test = np.zeros(5, dtype=complex)
J_test[0] = -1j*0.2165009384226491   # Q=-2
J_test[1] =  0.001530892816919044 -1j*0.0015308928169190817   # Q=-1
J_test[2] =  0.17678995321733607    # Q=0
J_test[3] = -0.001530892816919044 -1j*0.0015308928169190817   # Q=+1
J_test[4] =  1j*0.2165009384226491  # Q=+2

print(f"\nTest case J_vert:")
for Q in [-2, -1, 0, 1, 2]:
    if abs(J_test[Q+2]) > 1e-10:
        print(f"  J^2_{Q}: {J_test[Q+2]:.6f}")

print(f"\nDifference: test case has NON-ZERO ±2 components!")

# 05. 06. 2026. 21:22
Hu = 1e6

# --------------------------------------------------
# Radiation tensor in vertical frame
# --------------------------------------------------

w = 0.37890431177269185      # hR=0.1

Jvert = np.zeros(5, dtype=complex)
Jvert[2] = w/np.sqrt(2)

print("\nJ vertical")
for Q in [-2,-1,0,1,2]:
    print(Q, Jvert[Q+2])

# --------------------------------------------------
# Rotate to magnetic frame
# --------------------------------------------------

chiB = np.radians(45)
thetaB = np.pi/2

D = wigner_D2(chiB, thetaB, 0)

Jmag = D @ Jvert

print("\nJ magnetic")
for Q in [-2,-1,0,1,2]:
    print(Q, Jmag[Q+2])

# --------------------------------------------------
# Hanle operator
# --------------------------------------------------

rho_mag = np.zeros(5, dtype=complex)

for Q in [-2,-1,0,1,2]:

    rho_mag[Q+2] = (
        Jmag[Q+2]
        /
        (1 + 1j*Q*Hu)
    )

print("\nrho magnetic")
for Q in [-2,-1,0,1,2]:
    print(Q, rho_mag[Q+2])

# --------------------------------------------------
# Rotate back
# --------------------------------------------------

rho_vert = D.conj().T @ rho_mag

print("\nrho vertical")
for Q in [-2,-1,0,1,2]:
    print(Q, rho_vert[Q+2])

print(
    Jmag[2],
    -0.5*Jvert[2]
)

# 07. 06. 2026. 16:34
for i in [0,1,2]:
    print(f"\ni={i}")

    for K in [0,2]:
        for Q in range(-K,K+1):

            val = T(
                i=i,
                K=K,
                Q=Q,
                theta=0.0,
                chi=0.0,
                gamma=0.0
            )

            if abs(val) > 1e-12:
                print(K,Q,val)

# 07. 06. 2026. 17:15
from scipy.integrate import dblquad

sqrt2 = np.sqrt(2.0)

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


# ---------------------------------------------------------
# Radiation field tensor J^K_Q
# ---------------------------------------------------------

def JKQ(K, Q, hR):

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

    J[(0,0)] = JKQ(0,0,hR)

    for Q in [-2,-1,0,1,2]:
        J[(2,Q)] = JKQ(2,Q,hR)

    return J

hR = 0.1

J = radiation_tensor(hR)

print("J00 =", J[(0,0)])

for Q in [-2,-1,0,1,2]:
    print(Q, J[(2,Q)])

w_num = np.sqrt(2)*J[(2,0)]/J[(0,0)]

print()
print("w from tensor =", np.real(w_num))
print("w analytic    =", anisotropy_w2(hR))

# 07. 06. 2026. 17:28
sqrt3 = np.sqrt(3)

def idx(Q):
    return Q + 2

print("="*70)
print("FULL rho_{±2} SYMMETRY DIAGNOSTIC")
print("="*70)

Hu = 1e6
theta_B = np.pi/2

for chi_deg in [0, 30, 45, 60, 90]:

    print()
    print("="*70)
    print(f"chi_B = {chi_deg} deg")
    print("="*70)

    chi_B = np.radians(chi_deg)

    # --------------------------------------------------
    # Step 1
    # Vertical frame radiation tensor
    # --------------------------------------------------

    w = 0.3789043117726905

    Jvert = np.zeros(5,dtype=complex)
    Jvert[idx(0)] = w/np.sqrt(2)

    print()
    print("J vertical")
    for Q in [-2,-1,0,1,2]:
        print(Q, Jvert[idx(Q)])

    # --------------------------------------------------
    # Step 2
    # Rotate into magnetic frame
    # --------------------------------------------------

    D = wigner_D2(chi_B, theta_B, 0.0)

    Jmag = D @ Jvert

    print()
    print("J magnetic")
    for Q in [-2,-1,0,1,2]:
        print(Q, Jmag[idx(Q)])

    # --------------------------------------------------
    # Step 3
    # Hanle operator
    # --------------------------------------------------

    rho_mag = np.zeros(5,dtype=complex)

    for Q in [-2,-1,0,1,2]:

        rho_mag[idx(Q)] = (
            Jmag[idx(Q)]
            /
            (1 + 1j*Q*Hu)
        )

    print()
    print("rho magnetic")
    for Q in [-2,-1,0,1,2]:
        print(Q, rho_mag[idx(Q)])

    # --------------------------------------------------
    # Step 4
    # Rotate back
    # --------------------------------------------------

    rho_vert = D.conj().T @ rho_mag

    print()
    print("rho vertical")
    for Q in [-2,-1,0,1,2]:
        print(Q, rho_vert[idx(Q)])

    # --------------------------------------------------
    # Step 5
    # Symmetry analysis
    # --------------------------------------------------

    rm2 = rho_vert[idx(-2)]
    rp2 = rho_vert[idx(+2)]

    print()
    print("SYMMETRY TESTS")
    print("-"*40)

    print("rho_-2 =", rm2)
    print("rho_+2 =", rp2)

    print()

    print("rho_-2 - rho_+2")
    print(rm2-rp2)

    print()

    print("rho_-2 - conj(rho_+2)")
    print(rm2-np.conj(rp2))

    print()

    print("rho_-2 + rho_+2")
    print(rm2+rp2)

    print()

    print("|rho_-2-rho_+2| =",
          abs(rm2-rp2))

    print("|rho_-2-conj(rho_+2)| =",
          abs(rm2-np.conj(rp2)))

    # --------------------------------------------------
    # Step 6
    # Eq. (13.20)
    # theta = pi/2
    # chi = 0
    # gamma = 0
    # --------------------------------------------------

    epsQ_pm2 = (
        -sqrt3/2
        *
        (rm2 + rp2)
    )

    epsU_pm2 = (
        1j*sqrt3/2
        *
        (rm2 - rp2)
    )

    print()
    print("EXPLICIT ±2 CONTRIBUTIONS")
    print("-"*40)

    print("epsQ_pm2 =", epsQ_pm2)
    print("epsU_pm2 =", epsU_pm2)

    print()

    print("Re(epsQ_pm2) =", np.real(epsQ_pm2))
    print("Re(epsU_pm2) =", np.real(epsU_pm2))

    # --------------------------------------------------
    # Step 7
    # Full emissivity using T tensors
    # --------------------------------------------------

    epsQ = 0j
    epsU = 0j

    for Q in [-2,-1,0,1,2]:

        epsQ += (
            T(1,2,Q,
              np.pi/2,
              0.0,
              0.0)
            *
            rho_vert[idx(Q)]
        )

        epsU += (
            T(2,2,Q,
              np.pi/2,
              0.0,
              0.0)
            *
            rho_vert[idx(Q)]
        )

    print()
    print("FULL EMISSIVITY")
    print("-"*40)

    print("epsQ =", epsQ)
    print("epsU =", epsU)

    print()

    print("Re(epsQ) =", np.real(epsQ))
    print("Re(epsU) =", np.real(epsU))

# 07. 06. 2026. 17:39
print("Tensors at limb")

for Q in [-2,-1,0,1,2]:
    print(
        Q,
        "TQ =", T(1,2,Q,np.pi/2,0,0),
        "TU =", T(2,2,Q,np.pi/2,0,0)
    )

# 10. 06. 2026. Trying to solve SEE before applying Hanle effect
def tP(i, P):

    if i == 0:

        if P == 0:
            return 1/np.sqrt(2)

    elif i == 1:

        if P == -2:
            return -np.sqrt(3)/2

        if P == 2:
            return -np.sqrt(3)/2

    elif i == 2:

        if P == -2:
            return +1j*np.sqrt(3)/2

        if P == 2:
            return -1j*np.sqrt(3)/2

    return 0.0j


def T_book(i, Q,
           theta_obs,
           chi_obs,
           gamma_obs):

    Dobs = wigner_D2(
        chi_obs,
        theta_obs,
        gamma_obs
    )

    val = 0j

    for P in [-2,-1,0,1,2]:

        val += (
            tP(i,P)
            *
            Dobs[idx(P),idx(Q)]
        )

    return val

def rho_two_level_noB(JKQ):
    """
    Eq. (10.13)
    Jl=0 -> Ju=1
    w^(2)=1
    """

    rho = np.zeros(5,dtype=complex)

    for Q in [-2,-1,0,1,2]:

        rho[idx(Q)] = (
            (-1)**Q
            *
            JKQ[idx(-Q)]
        )

    return rho

def emissivity_two_level_noB(
        JKQ,
        theta_obs,
        chi_obs,
        gamma_obs):

    rho = rho_two_level_noB(JKQ)

    epsI = 0j
    epsQ = 0j
    epsU = 0j

    #
    # K=0 contribution
    #
    epsI += 1.0/np.sqrt(2)

    #
    # K=2 contribution
    #
    for Q in [-2,-1,0,1,2]:

        epsI += (
            T_book(
                0,Q,
                theta_obs,
                chi_obs,
                gamma_obs
            )
            * rho[idx(Q)]
        )

        epsQ += (
            T_book(
                1,Q,
                theta_obs,
                chi_obs,
                gamma_obs
            )
            * rho[idx(Q)]
        )

        epsU += (
            T_book(
                2,Q,
                theta_obs,
                chi_obs,
                gamma_obs
            )
            * rho[idx(Q)]
        )

    return epsI,epsQ,epsU

JKQ = np.zeros(5,dtype=complex)
JKQ[idx(0)] = 1.0

rho = rho_two_level_noB(JKQ)

print("rho")
for Q in [-2,-1,0,1,2]:
    print(Q, rho[idx(Q)])

epsI,epsQ,epsU = emissivity_two_level_noB(

    JKQ,

    theta_obs=np.pi/2,
    chi_obs=0.0,
    gamma_obs=0.0
)

print()
print("epsI =", epsI)
print("epsQ =", epsQ)
print("epsU =", epsU)

print()
print("Q/I =", np.real(epsQ/epsI))
print("U/I =", np.real(epsU/epsI))

# 10. 06. 2026. 19:08
print("="*70)
print("TENSOR TABLE")
print("="*70)

theta = np.pi/2
chi   = 0.0
gamma = 0.0

for i,name in zip([0,1,2],["I","Q","U"]):

    print()
    print(f"STOKES {name}")

    for Q in [-2,-1,0,1,2]:

        val = T_book(
            i,Q,
            theta,
            chi,
            gamma
        )

        print(
            f"Q={Q:+d}",
            val
        )

theta = np.pi/2
chi   = 0
gamma = 0
sqrt2 = np.sqrt(2)
sqrt3 = np.sqrt(3)

TI_ref = {
    -2 :  sqrt3/4,
    -1 :  0,
     0 : -1/(2*sqrt2),
     1 :  0,
     2 :  sqrt3/4
}

TQ_ref = {
    -2 :  sqrt3/4,
    -1 :  0,
     0 :  3/(2*sqrt2),
     1 :  0,
     2 :  sqrt3/4
}

TU_ref = {
    -2 : -1j*sqrt3/4,
    -1 : 0,
     0 : 0,
     1 : 0,
     2 : 1j*sqrt3/4
}

print()
print("="*70)
print("COMPARISON WITH LL04")
print("="*70)

for Q in [-2,-1,0,1,2]:

    TI = T_book(0,Q,np.pi/2,0,0)
    TQ = T_book(1,Q,np.pi/2,0,0)
    TU = T_book(2,Q,np.pi/2,0,0)

    print()
    print("Q =",Q)

    print(
        "I:",
        TI,
        "ref=",
        TI_ref[Q],
        "diff=",
        TI-TI_ref[Q]
    )

    print(
        "Q:",
        TQ,
        "ref=",
        TQ_ref[Q],
        "diff=",
        TQ-TQ_ref[Q]
    )

    print(
        "U:",
        TU,
        "ref=",
        TU_ref[Q],
        "diff=",
        TU-TU_ref[Q]
    )

rho = np.zeros(5,dtype=complex)
rho[idx(0)] = 1.0
epsI2 = T_book(0,0,np.pi/2,0,0)
epsQ2 = T_book(1,0,np.pi/2,0,0)

print("epsI2 =",epsI2)
print("epsQ2 =",epsQ2)

print("ratio =",epsQ2/epsI2)