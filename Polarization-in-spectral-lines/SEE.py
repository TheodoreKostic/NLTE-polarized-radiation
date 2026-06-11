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

def anisotropy_factor(Jrad):
    """
    Radiation anisotropy

    w = sqrt(2) J20/J00
    """

    return (
        np.sqrt(2.0)
        * np.real(Jrad[(2,0)] / Jrad[(0,0)])
    )

def Jrad_to_array(Jrad):

    Jarr = np.zeros(5, dtype=complex)

    for Q in [-2,-1,0,1,2]:
        Jarr[idx(Q)] = Jrad[(2,Q)]

    return Jarr

# with rho and J
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

    
    Jarr = Jrad_to_array(Jrad)

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
    J_hanle = H_full @ Jvert

    Hfull = D @ H_diag @ D.conj().T
    rho20 = Hfull @ Jarr

    # Compute emissivity
    epsI = rho00
    epsQ = 0.0j
    epsU = 0.0j

    for i, Q in enumerate(Qs):
        
        epsI += T(0, 2, Q, theta_obs, chi_obs, gamma_obs) * rho20[i]
        epsQ += T(1, 2, Q, theta_obs, chi_obs, gamma_obs) * rho20[i]
        epsU += T(2, 2, Q, theta_obs, chi_obs, gamma_obs) * rho20[i]

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

# Next step, try the full procedure

def J2_vertical(hR):

    Jrad = radiation_tensor(hR)

    Jvert = np.zeros(5,dtype=complex)

    for Q in [-2,-1,0,1,2]:
        Jvert[idx(Q)] = Jrad[(2,Q)]

    return Jvert

def J2_magnetic(
        hR,
        theta_B,
        chi_B):

    Jvert = J2_vertical(hR)

    D = wigner_D2(
        chi_B,
        theta_B,
        0.0
    )

    Jmag = D @ Jvert

    return Jvert, Jmag

def rho_vertical_from_magnetic(
        rho_mag,
        theta_B,
        chi_B):

    D = wigner_D2(
        chi_B,
        theta_B,
        0.0
    )

    rho_vert = D.conj().T @ rho_mag

    return rho_vert

def emissivity_from_rho(
        rho_vert,
        theta_obs,
        chi_obs,
        gamma_obs):

    epsI = 1.0 + 0j
    epsQ = 0j
    epsU = 0j

    for Q in [-2,-1,0,1,2]:

        coeff = (
            (-1)**Q
            *
            rho_vert[idx(-Q)]
        )

        epsI += (
            T(
                0,
                2,
                Q,
                theta_obs,
                chi_obs,
                gamma_obs
            )
            * coeff
        )

        epsQ += (
            T(
                1,
                2,
                Q,
                theta_obs,
                chi_obs,
                gamma_obs
            )
            * coeff
        )

        epsU += (
            T(
                2,
                2,
                Q,
                theta_obs,
                chi_obs,
                gamma_obs
            )
            * coeff
        )

    return epsI, epsQ, epsU

def hanle_two_level(
        hR,
        B,
        gJu,
        Aul,
        theta_B,
        chi_B,
        theta_obs,
        chi_obs,
        gamma_obs):

    #
    # Hu from Eq. (10.28)
    #
    Hu = hanle_parameter(
        B,
        gJu,
        Aul
    )

    #
    # J_Q in magnetic frame
    #
    Jvert, Jmag = J2_magnetic(
        hR,
        theta_B,
        chi_B
    )

    #
    # Eq. (10.27)
    #
    rho_mag = rhoKQ_hanle_two_level(
        Jmag,
        Hu
    )

    #
    # Rotate back
    #
    rho_vert = (
        wigner_D2(
            chi_B,
            theta_B,
            0.0
        ).conj().T
        @ rho_mag
    )

    #
    # Emissivity
    #
    epsI, epsQ, epsU = emissivity_from_rho(
        rho_vert,
        theta_obs,
        chi_obs,
        gamma_obs
    )

    return (
        epsI,
        epsQ,
        epsU,
        Jvert,
        Jmag,
        rho_mag,
        rho_vert
    )

B_s = [0.0, 1.0, 10.0, 100.0, 1000.0]

for B in B_s:
    epsI,epsQ,epsU,\
    Jvert,Jmag,\
    rho_mag,rho_vert = hanle_two_level(

        hR=0.1,

        B=B,
        gJu=1.0,
        Aul=1e8,

        theta_B=np.pi/3,
        chi_B=np.pi/4,

        theta_obs=np.pi/2,
        chi_obs=0.0,
        gamma_obs=0.0
    )
    print()
    print("B =", B)
    print("Q/I =", np.real(epsQ/epsI))
    print("U/I =", np.real(epsU/epsI))

# U/I is not zero for B = 0.0; what produces this stray polarization?

hR = 0.1

Jrad = radiation_tensor(hR)

J2 = np.zeros(5,dtype=complex)

for Q in [-2,-1,0,1,2]:
    J2[idx(Q)] = Jrad[(2,Q)]

print("Input radiation tensor")

for Q in [-2,-1,0,1,2]:
    print(Q, J2[idx(Q)])

print()
print("="*70)
print("HANLE SYMMETRY DIAGNOSTIC")
print("="*70)

for B in [0.0, 1.0, 10.0, 100.0, 1000.0]:

    Hu = hanle_parameter(B, gJu, Aul)

    rho = rhoKQ_hanle_two_level(
        J2,
        Hu
    )

    print()
    print(f"B = {B} G")
    print("-"*50)

    for Q in [-2,-1,0,1,2]:
        print(Q, rho[idx(Q)])

    print()
    print("Reality relations")

    print(
        "rho(-2)-conj(rho(+2)) =",
        rho[idx(-2)] - np.conj(rho[idx(2)])
    )

    print(
        "rho(-1)+conj(rho(+1)) =",
        rho[idx(-1)] + np.conj(rho[idx(1)])
    )

    print()
    print("Magnitudes")

    print("|rho0|  =", abs(rho[idx(0)]))
    print("|rho1|  =", abs(rho[idx(1)]))
    print("|rho2|  =", abs(rho[idx(2)]))

    if abs(rho[idx(0)]) > 0:
        print(
            "|rho1/rho0| =",
            abs(rho[idx(1)]/rho[idx(0)])
        )

        print(
            "|rho2/rho0| =",
            abs(rho[idx(2)]/rho[idx(0)])
        )

Bgrid = np.logspace(-2,3,200)

rho0 = []
rho1 = []
rho2 = []

for B in Bgrid:

    Hu = hanle_parameter(
        B,
        gJu,
        Aul
    )

    rho = rhoKQ_hanle_two_level(
        J2,
        Hu
    )

    rho0.append(abs(rho[idx(0)]))
    rho1.append(abs(rho[idx(1)]))
    rho2.append(abs(rho[idx(2)]))

plt.figure()

plt.loglog(Bgrid, rho0, label=r"$|\rho^2_0|$")
plt.loglog(Bgrid, rho1, label=r"$|\rho^2_1|$")
plt.loglog(Bgrid, rho2, label=r"$|\rho^2_2|$")

plt.xlabel("B [G]")
plt.ylabel("Density matrix amplitude")
plt.legend()
plt.grid(True)
plt.savefig("density_matrix_amp.png", dpi = 300)

# Apparently, the problem is in T
print()
print("="*60)
print("T^2_0 FOR STOKES U")
print("="*60)

for theta in [
    0,
    np.pi/6,
    np.pi/4,
    np.pi/3,
    np.pi/2
]:

    val = T(
        2,      # U
        2,
        0,
        theta,
        0.0,
        0.0
    )

    print(theta, val)

epsI = 0j
epsQ = 0j
epsU = 0j
theta_obs = np.pi/2
chi_obs = 0.0
gamma_obs = 0.0
rho0 = rho[idx(0)]

epsI += T(0,2,0,
          theta_obs,
          chi_obs,
          gamma_obs) * rho0

epsQ += T(1,2,0,
          theta_obs,
          chi_obs,
          gamma_obs) * rho0

epsU += T(2,2,0,
          theta_obs,
          chi_obs,
          gamma_obs) * rho0

print("epsI =", epsI)
print("epsQ =", epsQ)
print("epsU =", epsU)

# More testing
# ==========================================================
# FULL EMISSIVITY TEST
# ==========================================================

theta_obs = np.pi/2
chi_obs   = 0.0
gamma_obs = 0.0

# K=0 density matrix
rho00 = Jrad[(0,0)] / np.sqrt(3)

# K=2 density matrix
JKQ_array = np.zeros(5,dtype=complex)

for Q in [-2,-1,0,1,2]:
    JKQ_array[idx(Q)] = Jrad[(2,Q)]

rho20 = rhoKQ_hanle_two_level(
    JKQ_array,
    Hu
)

epsI = 0j
epsQ = 0j
epsU = 0j

# ----------------------------------------------------------
# K = 0 contribution
# ----------------------------------------------------------

epsI += (
    T(0,0,0,
      theta_obs,
      chi_obs,
      gamma_obs)
    * rho00
)

# ----------------------------------------------------------
# K = 2 contribution
# ----------------------------------------------------------

for Q in [-2,-1,0,1,2]:

    epsI += (
        T(0,2,Q,
          theta_obs,
          chi_obs,
          gamma_obs)
        * rho20[idx(Q)]
    )

    epsQ += (
        T(1,2,Q,
          theta_obs,
          chi_obs,
          gamma_obs)
        * rho20[idx(Q)]
    )

    epsU += (
        T(2,2,Q,
          theta_obs,
          chi_obs,
          gamma_obs)
        * rho20[idx(Q)]
    )

print()
print("epsI =", epsI)
print("epsQ =", epsQ)
print("epsU =", epsU)

print()
print("Q/I =", np.real(epsQ/epsI))
print("U/I =", np.real(epsU/epsI))

plt.figure()
plt.xlabel(r"$U/I$", fontsize=12)
plt.ylabel(r"$Q/I$", fontsize=12)
plt.plot(np.real(epsU/epsI), np.real(epsQ/epsI))
plt.title(r"Hanle diagram (full matrix operator, $\theta_B=90°$)", fontsize=12)
plt.grid(True, alpha=0.3)

plt.legend(fontsize=9, loc='best')
plt.tight_layout()
plt.savefig("Hanle_diagram.png", dpi = 300)


print()
print("rho00 =", rho00)

for Q in [-2,-1,0,1,2]:
    print(Q, rho20[idx(Q)])

# Cicles
print("="*60)
print("EMISSIVITY COEFFICIENT TEST")
print("="*60)

rho00 = rho00
rho20 = rho[idx(0)]

print("rho00 =", rho00)
print("rho20 =", rho20)
print("rho20/rho00 =", rho20/rho00)

mu = np.cos(theta_obs)

theory = (
    -(3.0/(2.0*np.sqrt(2.0)))
    *
    (1.0 - mu**2)
    *
    np.real(rho20/rho00)
)

print()
print("LL04 prediction =", theory)

print()
print("Tensor values")

TI00 = T(0,0,0,
          theta_obs,
          chi_obs,
          gamma_obs)

TI20 = T(0,2,0,
          theta_obs,
          chi_obs,
          gamma_obs)

TQ20 = T(1,2,0,
          theta_obs,
          chi_obs,
          gamma_obs)

print("T^0_0(I) =", TI00)
print("T^2_0(I) =", TI20)
print("T^2_0(Q) =", TQ20)

print()
print("Direct emissivity ratio")

epsI_test = (
    TI00*rho00
    +
    TI20*rho20
)

epsQ_test = (
    TQ20*rho20
)

print("epsI_test =", epsI_test)
print("epsQ_test =", epsQ_test)

print("Q/I from tensors =",
      np.real(epsQ_test/epsI_test))

alignment = np.sqrt(2.0) * rho[idx(0)] / rho00

print("alignment =", alignment)

# ============================================================
# TESTS
# ============================================================

hR = 0.1
gamma_obs = np.pi/2
Jrad = radiation_tensor(hR)

print("w =", anisotropy_factor(Jrad))

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
            J_rad = Jrad,
            theta_B=np.pi/2,
            chi_B=np.radians(chi),
            theta_obs=np.pi/2,
            chi_obs=0.0,
            gamma_obs=gamma_obs
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
        J_rad=Jrad,
        theta_B=np.pi/2,
        chi_B=chi_B,
        theta_obs=np.pi/2,
        chi_obs=0.0,
        gamma_obs=gamma_obs
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

# ============================================================
# PLOT HANLE DIAGRAM
# ============================================================

print()
print()
print("GENERATING PLOT")
print("-" * 70)

Hu_values = [0.08, 0.16, 0.25, 0.36,
             0.50, 0.69, 0.98,
             1.54, 3.16, 1e6]

chi_B_grid = np.linspace(0, np.pi, 721)

plt.figure(figsize=(9, 9))

# ============================================================
# DASHED CURVES : Hu = const
# ============================================================

label_fraction = {
    0.08: 0.88,
    0.16: 0.87,
    0.25: 0.86,
    0.36: 0.84,
    0.50: 0.82,
    0.69: 0.80,
    0.98: 0.77,
    1.54: 0.72,
    3.16: 0.65,
}

for Hu in Hu_values:

    PU = []
    PQ = []

    for chi_B in chi_B_grid:

        pQ, pU = hanle_polarization_corrected(
            Hu=Hu,
            J_rad=Jrad,
            theta_B=np.pi/2,
            chi_B=chi_B,
            theta_obs=np.pi/2,
            chi_obs=0.0,
            gamma_obs=gamma_obs
        )

        PU.append(pU)
        PQ.append(pQ)

    PU = np.array(PU)
    PQ = np.array(PQ)

    plt.plot(
        PU,
        PQ,
        '--',
        lw=1.5
    )

    # Label Hu curves

    if Hu < 1e6:

        idx_lab = int(
            label_fraction[Hu] * len(PU)
        )

        plt.text(
            PU[idx_lab],
            PQ[idx_lab],
            f"{Hu:g}",
            fontsize=9,
            ha='left',
            va='center',
            bbox=dict(
                facecolor='white',
                edgecolor='none',
                alpha=0.8
            )
        )

# ============================================================
# SOLID CURVES : chi_B = const
# ============================================================

chi_const_deg = [0, 30, 60, 90, 120, 150, 180]

Hu_grid = np.logspace(
    np.log10(0.08),
    np.log10(1e6),
    400
)

for chi_deg in chi_const_deg:

    PU = []
    PQ = []

    for Hu in Hu_grid:

        pQ, pU = hanle_polarization_corrected(
            Hu=Hu,
            J_rad=Jrad,
            theta_B=np.pi/2,
            chi_B=np.radians(chi_deg),
            theta_obs=np.pi/2,
            chi_obs=0.0,
            gamma_obs=gamma_obs
        )

        PU.append(pU)
        PQ.append(pQ)

    PU = np.array(PU)
    PQ = np.array(PQ)

    # Horizontal flip to match Fig. 13.3

    PU_plot = PU

    plt.plot(
        PU_plot,
        PQ,
        '-',
        lw=1.0
    )

    label_Hu = {
        0:   3.16,
        30:  1.54,
        60:  0.69,
        90:  0.36,
        120: 0.69,
        150: 1.54,
        180: 3.16,
    }

    idx_lab = np.argmin(
        np.abs(Hu_grid - label_Hu[chi_deg])
    )

    plt.text(
        PU_plot[idx_lab],
        PQ[idx_lab],
        f"{chi_deg}°",
        fontsize=8,
        ha='center',
        va='center',
        bbox=dict(
            facecolor='white',
            edgecolor='none',
            alpha=0.8
        )
    )

# ============================================================
# SATURATED LIMIT MARKERS
# ============================================================
'''
Hu = 1e6

for chi_deg in [0, 45, 90, 135, 180]:

    pQ, pU = hanle_polarization_corrected(
        Hu=Hu,
        J_rad=Jrad,
        theta_B=np.pi/2,
        chi_B=np.radians(chi_deg),
        theta_obs=np.pi/2,
        chi_obs=0.0,
        gamma_obs=gamma_obs
    )

    plt.plot(pU, pQ, 'ko', markersize=5)

    plt.text(
        pU + 0.01,
        pQ,
        f"{chi_deg}°",
        fontsize=9
    )
'''
plt.xlabel(r"$U/I$", fontsize=12)
plt.ylabel(r"$Q/I$", fontsize=12)

plt.title(
    r"Hanle diagram ($\theta_B=90^\circ$)",
    fontsize=12
)

plt.grid(True, alpha=0.3)
plt.axis("equal")

plt.tight_layout()

plt.savefig(
    "Hanle_substitute.png",
    dpi=300
)

print("Saved: Hanle_substitute.png")
'''
# Mark chi_B on saturated curve
Hu = 1e6
for chi_deg in [0, 45, 90, 135, 180]:
    pQ, pU = hanle_polarization_corrected(
        Hu=Hu,
        J_rad = Jrad,
        theta_B=np.pi/2,
        chi_B=np.radians(chi_deg),
        theta_obs=np.pi/2,
        chi_obs=0.0,
        gamma_obs=np.pi/2
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
plt.savefig("Hanle_substitute.png", dpi=300)
print("Saved: Hanle_substitute.png")
'''
for Hu in [0.08,0.16,0.25,0.36,0.5,0.69,0.98,1.54,3.16]:
    pQ,pU = hanle_polarization_corrected(
        Hu=Hu,
        J_rad=Jrad,
        theta_B=np.pi/2,
        chi_B=np.radians(30),
        theta_obs=np.pi/2,
        chi_obs=0,
        gamma_obs=np.pi/2
    )

    print(Hu,pQ,pU)


for chi_deg in [0,30,60,90,120,150,180]:

    pQ,pU = hanle_polarization_corrected(
        Hu=0.69,
        J_rad=Jrad,
        theta_B=np.pi/2,
        chi_B=np.radians(chi_deg),
        theta_obs=np.pi/2,
        chi_obs=0.0,
        gamma_obs=gamma_obs
    )

    print(chi_deg, pQ, pU)

for chi_deg in [0,30,60,90,120,150,180]:
    pQ, pU = hanle_polarization_corrected(
        Hu=1e6,
        J_rad=Jrad,
        theta_B=np.pi/2,
        chi_B=np.radians(chi_deg),
        theta_obs=np.pi/2,
        chi_obs=0,
        gamma_obs=np.pi/2
    )

    print(
        chi_deg,
        "Q=", pQ,
        "U=", pU
    )

Jarr = Jrad_to_array(Jrad)

D = wigner_D2(0.0, np.pi/2, 0.0)

Jmag = D.conj().T @ Jarr

for q,val in zip([-2,-1,0,1,2], Jmag):
    print(q, val)


D = wigner_D2(0.0, np.pi/2, 0.0)

Jmag = D.conj().T @ Jarr

for q,val in zip([-2,-1,0,1,2], Jmag):
    print(q, val)