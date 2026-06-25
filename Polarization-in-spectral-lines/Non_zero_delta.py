import numpy as np
import sys
import os
import matplotlib.pyplot as plt

script_dir = os.path.abspath("/home/Code/NLTE-polarized-radiation")
#script_dir = os.path.abspath("/home/teodor/Documents/Codes/NLTE-polarized-radiation")
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
    # I
    if i == 0:
        if K == 0 and Q == 0:
            return 1.0
        if K == 2 and Q == 0:
            return (3*ct**2 - 1)/(2*sqrt2)
        if K == 2 and Q == 1:
            return -sqrt3/2 * st*ct * ex1
        if K == 2 and Q == 2:
            return sqrt3/4 * st**2 * ex2
    # Q
    if i == 1:
        if K == 2 and Q == 0:
            return -(3/(2*sqrt2))*st**2 * c2
        if K == 2 and Q == 1:
            return -(sqrt3/2) * (c2*ct + 1j*s2) * st * ex1
        if K == 2 and Q == 2:
            return -(sqrt3/4) * (c2*(1+ct**2) + 2j*s2*ct) * ex2
    # U
    if i == 2:
        if K == 2 and Q == 0:
            return (3/(2*sqrt2))*s2*st**2
        if K == 2 and Q == 1:
            return (sqrt3/2) * (s2*ct - 1j*c2) * st * ex1
        if K == 2 and Q == 2:
            return (sqrt3/4) * (s2*(1+ct**2) - 2j*c2*ct) * ex2
    # V
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

def qidx(Q):
    return Q + 2

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
    Jarr = Jrad_to_array(J_rad)
   
    rho00 = Jrad[(0,0)]
    
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
    
    #D_conj_T = D.conj().T
    #H_full = D @ H_diag @ D_conj_T
    
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
    #print("I = ", epsI)
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

def radiation_tensor_delta(hR, delta):

    J0 = radiation_tensor(hR)

    Jarr = np.zeros(5, dtype=complex)

    Jarr[idx(0)] = J0[(2,0)]

    D = wigner_D2(0.0, 0.0 + delta, 0.0)

    Jrot = D @ Jarr

    J = {}

    J[(0,0)] = J0[(0,0)]

    for Q in [-2,-1,0,1,2]:
        J[(2,Q)] = Jrot[idx(Q)]

    return J

print("---------")
delta = np.radians(30.0)
hp = 0.073      # projected height from Fig. 13.3
hR = (1 + hp)/np.cos(delta) - 1
Jrad = radiation_tensor_delta(hR, np.radians(30))
J0 = radiation_tensor_delta(hR, np.radians(0))
J30 = radiation_tensor_delta(hR, np.radians(30))
Jm30 = radiation_tensor_delta(hR, np.radians(-30.0))
S0 = sum(abs(J0[(2,Q)])**2 for Q in range(-2,3))
S30 = sum(abs(J30[(2,Q)])**2 for Q in range(-2,3))
print(S0)
print(S30)

print("---------")
for Hu in [1e-6,1e-4,1e-2,0.1,1.0,10,1e6]:
    for chi_B in [0, 90, 270]:
        pQ,pU = hanle_polarization_corrected(
            Hu=Hu,
            J_rad=Jrad,
            theta_B=np.pi/2,
            chi_B=np.radians(chi_B),
            theta_obs=np.pi/2,
            chi_obs=0.0,
            gamma_obs=np.pi/2
        )
        print(Hu,chi_B,pQ,pU)
plt.figure()
for chi in np.arange(0,360,30):
    pQ,pU = hanle_polarization_corrected(
        Hu=1e6,
        J_rad=Jrad,
        theta_B=np.pi/2,
        chi_B=np.radians(chi),
        theta_obs=np.pi/2,
        chi_obs=0.0,
        gamma_obs=np.pi/2
    )
    plt.plot(pU,pQ,'ko')
    plt.text(pU,pQ,f"{chi}")
plt.savefig("Fig_13_4_test.png", dpi = 300)

for delta in [+30, -30]:
    J = radiation_tensor_delta(hR, np.radians(delta))
    print(delta)
    for Q in [-2,-1,0,1,2]:
        print(Q, J[(2,Q)])

print("-----------------------")
for chi in [15, 37, 73, 111]:

    pQ1,pU1 = hanle_polarization_corrected(
        Hu=0.5,
        J_rad=J30,
        theta_B=np.pi/2,
        chi_B=np.radians(chi),
        theta_obs=np.pi/2,
        chi_obs=0.0,
        gamma_obs=np.pi/2
    )

    pQ2,pU2 = hanle_polarization_corrected(
        Hu=0.5,
        J_rad=Jm30,
        theta_B=np.pi/2,
        chi_B=np.radians(chi),
        theta_obs=np.pi/2,
        chi_obs=0.0,
        gamma_obs=np.pi/2
    )

    print()
    print("chi =",chi)
    print(" +30 :",pQ1,pU1)
    print(" -30 :",pQ2,pU2)
    print(" diff:",pQ1-pQ2,pU1-pU2)
print("-------------------")
Hu      = 0.5
theta_B = np.pi/2
chi_B   = np.radians(37)
theta_obs = np.pi/2
chi_obs   = 0.0
gamma_obs = np.pi/2

Jarr = Jrad_to_array(J30)

D = wigner_D2(chi_B, theta_B, 0.0)

Qs = np.array([-2,-1,0,1,2])

H_diag = np.diag(
    [1/(1+1j*Q*Hu) for Q in Qs]
)

H_full = D @ H_diag @ D.conj().T

rho20 = H_full @ Jarr

print()
print("rho20:")
for Q in Qs:
    print(Q, rho20[idx(Q)])

epsQ = 0j
epsU = 0j

print("\nContributions:")
print("--------------------------------")

for i,Q in enumerate(Qs):

    rho = rho20[idx(-Q)]

    phase = (-1.0)**Q

    termQ = (
        phase
        * T(1,2,Q,
            theta_obs,
            chi_obs,
            gamma_obs)
        * rho
    )

    termU = (
        phase
        * T(2,2,Q,
            theta_obs,
            chi_obs,
            gamma_obs)
        * rho
    )

    print(
        f"Q={Q:+d}",
        "rho=",rho,
        "termQ=",termQ,
        "termU=",termU
    )

    epsQ += termQ
    epsU += termU

print()
print("epsQ =",epsQ)
print("epsU =",epsU)

print("---------------------")
Jarr = Jrad_to_array(Jm30)

D = wigner_D2(chi_B, theta_B, 0.0)

Qs = np.array([-2,-1,0,1,2])

H_diag = np.diag(
    [1/(1+1j*Q*Hu) for Q in Qs]
)

H_full = D @ H_diag @ D.conj().T

rho20 = H_full @ Jarr

print()
print("rho20:")
for Q in Qs:
    print(Q, rho20[idx(Q)])

epsQ = 0j
epsU = 0j

print("\nContributions:")
print("--------------------------------")

for i,Q in enumerate(Qs):

    rho = rho20[idx(-Q)]

    phase = (-1.0)**Q

    termQ = (
        phase
        * T(1,2,Q,
            theta_obs,
            chi_obs,
            gamma_obs)
        * rho
    )

    termU = (
        phase
        * T(2,2,Q,
            theta_obs,
            chi_obs,
            gamma_obs)
        * rho
    )

    print(
        f"Q={Q:+d}",
        "rho=",rho,
        "termQ=",termQ,
        "termU=",termU
    )

    epsQ += termQ
    epsU += termU

print()
print("epsQ =",epsQ)
print("epsU =",epsU)


print("------------------")
print("\n===== J30 =====")

q, u = hanle_polarization_corrected(
    Hu=0.5,
    J_rad=J30,
    theta_B=np.pi/2,
    chi_B=np.radians(37),
    theta_obs=np.pi/2,
    chi_obs=0.0,
    gamma_obs=np.pi/2
)
print(q,u)

print("\n===== Jm30 =====")

q, u = hanle_polarization_corrected(
    Hu=0.5,
    J_rad=Jm30,
    theta_B=np.pi/2,
    chi_B=np.radians(37),
    theta_obs=np.pi/2,
    chi_obs=0.0,
    gamma_obs=np.pi/2
)
print(q,u)

print("----------------------")
J30  = radiation_tensor_delta(hR, np.radians(30))
Jm30 = radiation_tensor_delta(hR, np.radians(-30))

print("\nJ30")
for Q in [-2,-1,0,1,2]:
    print(Q, J30[(2,Q)])

print("\nJm30")
for Q in [-2,-1,0,1,2]:
    print(Q, Jm30[(2,Q)])

theta_obs = np.pi/2
chi_B = 0.0
gamma_obs = np.pi/2
J30 = radiation_tensor_delta(hR, np.radians(30))
Jm30 = radiation_tensor_delta(hR, np.radians(-30))
for name,J in [("J30",J30),("Jm30",Jm30)]:

    print()
    print("=====",name,"=====")

    Jarr = Jrad_to_array(J)

    rho20 = H_full @ Jarr

    for Q in [-2,-1,0,1,2]:
        print(Q, rho20[idx(Q)])

    epsQ = 0j
    epsU = 0j

    for Q in [-2,-1,0,1,2]:

        rho = rho20[idx(-Q)]

        epsQ += (
            (-1)**Q
            * T(1,2,Q,
                theta_obs,
                chi_obs,
                gamma_obs)
            * rho
        )

        epsU += (
            (-1)**Q
            * T(2,2,Q,
                theta_obs,
                chi_obs,
                gamma_obs)
            * rho
        )

    print("epsQ =",epsQ)
    print("epsU =",epsU)

plt.figure()
Hu_grid = np.logspace(-6, 2, 200)

Q30 = []
U30 = []

Qm30 = []
Um30 = []

for Hu in Hu_grid:

    q,u = hanle_polarization_corrected(
        Hu,
        J30,
        np.pi/2,
        np.radians(37),
        np.pi/2,
        0.0,
        np.pi/2
    )

    Q30.append(q)
    U30.append(u)

    q,u = hanle_polarization_corrected(
        Hu,
        Jm30,
        np.pi/2,
        np.radians(37),
        np.pi/2,
        0.0,
        np.pi/2
    )

    Qm30.append(q)
    Um30.append(u)
plt.plot(U30,Q30,label="+30")
plt.plot(Um30,Qm30,label="-30")
plt.legend()
plt.savefig("Test_plusminus.png", dpi = 300)


# Fig. 13.4, delta = 30 deg
delta = np.radians(30.0)
hp = 0.073      # projected height from Fig. 13.3
hR = (1 + hp)/np.cos(delta) - 1
gamma_obs = np.pi/2
chi_obs = 0.0

Jrad = radiation_tensor_delta(hR, np.radians(30))

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
            chi_obs=chi_obs,
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
        chi_obs=chi_obs,
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

Hu_values = [0.01, 0.08, 0.16, 0.25, 0.36,
             0.50, 0.69, 0.98,
             1.54, 3.16]

chi_B_grid = np.linspace(0, 2*np.pi, 721)

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
            chi_obs=chi_obs,
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
        lw=1.5,
        label=f"Hu={Hu:g}"
    )

    # Label Hu curves
    '''
    if Hu in label_fraction:

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
        ) '''
    plt.legend(
    loc='upper right',
    fontsize=8
    )

# ============================================================
# SOLID CURVES : chi_B = const
# ============================================================

chi_const_deg = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]

Hu_grid = np.logspace(-6, np.log10(3.16), 400)

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
            chi_obs=chi_obs,
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
        'k-',
        lw=1.0
    )

    curve_number = {
    0: 1,
    30: 2,
    60: 3,
    90: 4,
    120: 5,
    150: 6,
    180: 7,
    210: 8,
    240: 9,
    270: 10,
    300: 11,
    330: 12,
}

    idx_lab = np.argmin(np.abs(Hu_grid - curve_number[chi_deg]))

    plt.text(
        PU_plot[idx_lab],
        PQ[idx_lab],
        f"{curve_number[chi_deg]}",
        fontsize=8,
        ha='center',
        va='center',
        bbox=dict(
            facecolor='white',
            edgecolor='none',
            alpha=0.8
        )
    )

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
    "Hanle_substitute_30deg.png",
    dpi=300
)
print("Saved: Hanle_substitute_30deg.png")

print("2x2 Diagnosti panel, 30deg delta")
# ============================================================
# 2x2 DIAGNOSTIC PANEL
# ============================================================

fig, axes = plt.subplots(
    2, 2,
    figsize=(14, 14)
)

observer_setups = [

    (0.0, np.pi/2,
     r"$\chi_{\rm obs}=0,\ \gamma_{\rm obs}=90^\circ$"),

    (np.pi/2, np.pi/2,
     r"$\chi_{\rm obs}=90^\circ,\ \gamma_{\rm obs}=90^\circ$"),

    (0.0, 0.0,
     r"$\chi_{\rm obs}=0,\ \gamma_{\rm obs}=0^\circ$"),

    (np.pi/2, 0.0,
     r"$\chi_{\rm obs}=90^\circ,\ \gamma_{\rm obs}=0^\circ$")
]

for ax, (chi_obs, gamma_obs, panel_title) in zip(
        axes.ravel(),
        observer_setups):

    # ========================================================
    # DASHED CURVES : Hu = const
    # ========================================================

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
                chi_obs=chi_obs,
                gamma_obs=gamma_obs
            )

            PU.append(pU)
            PQ.append(pQ)

        PU = np.array(PU)
        PQ = np.array(PQ)

        ax.plot(
            PU,
            PQ,
            '--',
            lw=1.5,
            label=f"Hu={Hu:g}"
        )

        if Hu in label_fraction:

            idx_lab = int(
                label_fraction[Hu] * len(PU)
            )
            '''
            ax.text(
                PU[idx_lab],
                PQ[idx_lab],
                f"{Hu:g}",
                fontsize=8,
                bbox=dict(
                    facecolor='white',
                    edgecolor='none',
                    alpha=0.8
                )
            )'''
        ax.legend(
        loc='upper right',
        fontsize=8
        )

    # ========================================================
    # SOLID CURVES : chiB = const
    # ========================================================

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
                chi_obs=chi_obs,
                gamma_obs=gamma_obs
            )

            PU.append(pU)
            PQ.append(pQ)

        PU = np.array(PU)
        PQ = np.array(PQ)

        ax.plot(
            PU,
            PQ,
            '-',
            lw=1.2
        )

        label_Hu = {
        0:   1,
        30:  2,
        60:  3,
        90:  4,
        120: 5,
        150: 6,
        180: 7,
        210: 8,
        240: 9,
        270: 10,
        300: 11,
        330: 12
        }

        idx_lab = np.argmin(
            np.abs(
                Hu_grid
                - label_Hu[chi_deg]
            )
        )

        ax.text(
            PU[idx_lab],
            PQ[idx_lab],
            f"{chi_deg}°",
            fontsize=8,
            bbox=dict(
                facecolor='white',
                edgecolor='none',
                alpha=0.8
            )
        )

    ax.set_title(panel_title)

    ax.set_xlabel(r"$U/I$")
    ax.set_ylabel(r"$Q/I$")

    ax.grid(True, alpha=0.3)

    ax.set_aspect("equal")

plt.suptitle(
    r"Hanle diagram diagnostic ($\theta_B=90^\circ$)",
    fontsize=16
)

plt.tight_layout()

plt.savefig(
    "Hanle_2x2_diagnostic_30deg.png",
    dpi=300
)

# ==========================================================
# DELTA COMPARISON
# ==========================================================

deltas = {
    "0°"   : ("k", 0.0),
    "+30°" : ("r", np.radians(30)),
    "-30°" : ("b", np.radians(-30)),
}

Hu_values = [
    0.08, 0.16, 0.25, 0.36,
    0.50, 0.69, 0.98,
    1.54, 3.16
]

chi_const_deg = [
    0,30,60,90,120,150,
    180,210,240,270,300,330
]

chi_B_grid = np.linspace(0, 2*np.pi, 721)
Hu_grid = np.logspace(-6, np.log10(3.16), 400)

plt.figure(figsize=(10,10))

for label,(color,delta) in deltas.items():

    hp = 0.073
    hR = (1 + hp)/np.cos(delta) - 1

    Jrad = radiation_tensor_delta(hR, delta)

    # ---------------------------------------------
    # Dashed family: Hu = const
    # ---------------------------------------------

    for Hu in Hu_values:

        PU = []
        PQ = []

        for chi_B in chi_B_grid:

            pQ,pU = hanle_polarization_corrected(
                Hu=Hu,
                J_rad=Jrad,
                theta_B=np.pi/2,
                chi_B=chi_B,
                theta_obs=np.pi/2,
                chi_obs=0.0,
                gamma_obs=np.pi/2
            )

            PU.append(pU)
            PQ.append(pQ)

        plt.plot(
            PU,
            PQ,
            '--',
            color=color,
            alpha=0.6
        )

    # ---------------------------------------------
    # Solid family: chiB = const
    # ---------------------------------------------

    for chi_deg in chi_const_deg:

        PU = []
        PQ = []

        for Hu in Hu_grid:

            pQ,pU = hanle_polarization_corrected(
                Hu=Hu,
                J_rad=Jrad,
                theta_B=np.pi/2,
                chi_B=np.radians(chi_deg),
                theta_obs=np.pi/2,
                chi_obs=0.0,
                gamma_obs=np.pi/2
            )

            PU.append(pU)
            PQ.append(pQ)

        plt.plot(
            PU,
            PQ,
            '-',
            color=color,
            alpha=0.8
        )

# dummy legend entries
plt.plot([],[],'k-',label=r'$\delta=0^\circ$')
plt.plot([],[],'r-',label=r'$\delta=+30^\circ$')
plt.plot([],[],'b-',label=r'$\delta=-30^\circ$')

plt.xlabel("U/I")
plt.ylabel("Q/I")
plt.legend()
plt.grid(True,alpha=0.3)
plt.axis("equal")
plt.tight_layout()
plt.savefig("Delta_mdelta_zero.png", dpi = 300)

# J testing
delta = np.radians(30.0)
hp = 0.073      # projected height from Fig. 13.3
hR = (1 + hp)/np.cos(delta) - 1
gamma_obs = np.pi/2
chi_obs = 0.0
Jrad = radiation_tensor_delta(hR, np.radians(30))
for Q in [-2,-1,0,1,2]:
    print(
        Q,
        Jrad[(2,Q)] / Jrad[(0,0)]
    )

Hu_test = np.logspace(-3,1,200)

Q = []

for Hu in Hu_test:

    pQ,pU = hanle_polarization_corrected(
        Hu=Hu,
        J_rad=Jrad,
        theta_B=np.pi/2,
        chi_B=np.radians(270),
        theta_obs=np.pi/2,
        chi_obs=0.0,
        gamma_obs=np.pi/2
    )

    Q.append(pQ)

Q = np.array(Q)

print("max Q/I =", Q.max())
print("at Hu =", Hu_test[np.argmax(Q)])
print("Q(H=0) =", Q[0])

Jrad = radiation_tensor(hR)
print("J00 =", Jrad[(0,0)])
print("J20 =", Jrad[(2,0)])
print("w =", anisotropy_factor(Jrad))

Jrad = radiation_tensor_delta(hR, np.radians(30))
print("J00 =", Jrad[(0,0)])
print("J20 =", Jrad[(2,0)])
print("w =", anisotropy_factor(Jrad))

print("NEW NUMERICAL TEST FOR CERTAIN FAMILIES")
for Hu in [0,0.1,0.3,1,3]:
    pQ,pU = hanle_polarization_corrected(
        Hu=Hu,
        J_rad=Jrad,
        theta_B=np.pi/2,
        chi_B=np.pi/2,
        theta_obs=np.pi/2,
        chi_obs=0.0,
        gamma_obs=np.pi/2
    )
    print(Hu,pQ,pU)
print("CHI_B = 270")
for Hu in [0,0.1,0.3,1,3]:
    pQ,pU = hanle_polarization_corrected(
        Hu=Hu,
        J_rad=Jrad,
        theta_B=np.pi/2,
        chi_B=3*np.pi/2,
        theta_obs=np.pi/2,
        chi_obs=0.0,
        gamma_obs=np.pi/2
    )
    print(Hu,pQ,pU)
print("HU = 0.98")
for chi in [0,30,60,90,120,150,180]:
    pQ,pU = hanle_polarization_corrected(
        Hu=0.98,
        J_rad=Jrad,
        theta_B=np.pi/2,
        chi_B=chi,
        theta_obs=np.pi/2,
        chi_obs=0.0,
        gamma_obs=np.pi/2
    )
    print(chi,pQ,pU)

print("CONVENVTION")
Hu_grid = np.linspace(0,1,200)

Qvals = []

for Hu in Hu_grid:
    pQ,pU = hanle_polarization_corrected(
        Hu=Hu,
        J_rad=Jrad,
        theta_B=np.pi/2,
        chi_B=np.radians(-90),
        theta_obs=np.pi/2,
        chi_obs=0.0,
        gamma_obs=np.pi/2
    )

    Qvals.append(pQ)

imax = np.argmax(Qvals)

print(Hu_grid[imax], Qvals[imax])
print("Q(H=0) =", Qvals[0])


# Moving on to Fig 13.6
print("+++++++++++++++++++++++++++++Stokes profiles+++++++++++++++++++++++++++++")
theta_B = np.radians(90)
chi_B = 0.0
theta_obs=np.pi/2
chi_obs=0.0
gamma_obs=np.pi/2

Jrad_0 = radiation_tensor(hR=0.073)

def doppler_profile(x):
    """
    Normalized Doppler profile.

    x = (nu - nu0)/Delta_nu_D
    """
    return np.exp(-x**2) / np.sqrt(np.pi)

# def optical depth
# S_I, S_Q, S_U
# 1. Polje zracenja definisem za datu opticku dubinu (potrebno je resiti JPZ)
# 2. J-otovi se odrede 
# 3. Iz J i B dobiti S   
# * profili oblika iz (10.40) za dati prelaz
# 4. Iz S dobiti novo polje zracenja ili Stoksove paramtere u zeljenom pravcu i na zeljenim talasnim duzinama
# 

def stokes_I_profile(
        x_grid,
        Hu,
        J_rad,
        theta_B,
        chi_B,
        theta_obs,
        chi_obs,
        gamma_obs=np.pi/2):

    Jarr = Jrad_to_array(J_rad)
    rho00 = J_rad[(0,0)]
    Qs = np.array([-2, -1, 0, 1, 2])

    H_diag = np.diag(
        [1.0/(1.0 + 1j*Q*Hu) for Q in Qs]
    )

    D = wigner_D2(chi_B, theta_B, 0.0)

    Hfull = D @ H_diag @ D.conj().T

    rho = Hfull @ Jarr

    I_profile = []

    for x in x_grid:

        phi = doppler_profile(x)

        epsI = rho00

        for Q in Qs:

            term = (
                (-1.0)**Q
                * T(0, 2, Q,
                    theta_obs,
                    chi_obs,
                    gamma_obs)
                * rho[idx(-Q)]
            )

            epsI += phi * term

        I_profile.append(np.real(epsI))

    return np.array(I_profile)

x = np.linspace(-5, 5, 401)
Jrad_0 = radiation_tensor(hR=0.073)
I = stokes_I_profile(
        x,
        Hu=1.0,
        J_rad=Jrad_0,
        theta_B=np.radians(90),
        chi_B=0.0,
        theta_obs=np.pi/2,
        chi_obs=0.0)
plt.figure()
plt.plot(x, I)
plt.xlabel(r'$(\nu-\nu_0)/\Delta\nu_D$')
plt.ylabel('I')
plt.savefig("Mock_StokesI.png", dpi = 300)
print(I)


Jarr = Jrad_to_array(Jrad_0)

Qs = np.array([-2,-1,0,1,2])

H_diag = np.diag([1/(1+1j*Q*1.0) for Q in Qs])

D = wigner_D2(0.0, np.pi/2, 0.0)

rho20 = D @ H_diag @ D.conj().T @ Jarr

epsI = Jrad_0[(0,0)]

for Q in Qs:
    rho = rho20[idx(-Q)]

    epsI += (
        (-1.0)**Q
        * T(0,2,Q,
            np.pi/2,
            0.0,
            np.pi/2)
        * rho
    )

print("epsI =", epsI)

epsI0 = Jrad_0[(0,0)]

epsI2 = 0.0j

for Q in Qs:
    rho = rho20[idx(-Q)]

    epsI2 += (
        (-1.0)**Q
        * T(0,2,Q,
            theta_obs,
            chi_obs,
            gamma_obs)
        * rho
    )

print("epsI0 =", epsI0)
print("epsI2 =", epsI2)
print("epsI total =", epsI0 + epsI2)

# Testing profile from the book

from Profile_fun import *
from Hanle_fun import *
from Radiation_fun import *
from sympy.physics.wigner import wigner_3j
x = np.linspace(-5,5,401)

P00 = Phi_generalized(
    x,
    K=0,
    Kp=0,
    Q=0,
    vH=1.0
)

P20 = Phi_generalized(
    x,
    K=2,
    Kp=0,
    Q=0,
    vH=1.0
)

P22 = Phi_generalized(
    x,
    K=2,
    Kp=2,
    Q=0,
    vH=1.0
)

plt.figure()
plt.xlabel("Reduced wavelength")
plt.ylabel("Phi")
plt.plot(x,P00, label = "P00")
plt.plot(x,P20, label = "P20")
plt.plot(x,P22, label = "P22")
plt.legend()
plt.savefig("Mock_profile.png", dpi = 300)
print(P00.min(), P00.max())

vH = 1.0
phi_sum = (
    phi_doppler(x + vH)
    + phi_doppler(x)
    + phi_doppler(x - vH)
)

phi_sum /= phi_sum.max()

P00n = P00 / P00.max()

plt.figure()
plt.plot(x, P00n, label="P00")
plt.plot(x, phi_sum, "--", label="sum of Zeeman profiles")
plt.legend()
plt.savefig("Mock_comparison.png", dpi = 300)

print("---------------------")
for K in [0,2]:
    for Kp in [0,2]:
        for Q in [-2,-1,0,1,2]:
            val = np.max(
                np.abs(
                    Phi_generalized(
                        x,K,Kp,Q,vH=1.0
                    )
                )
            )
            print(K,Kp,Q,val)

print("---------------------")
for K in [0, 2]:
    for Kp in [0, 2]:
        for vH in [0.0, 0.5, 2.0]:

            P = Phi_generalized(
                x,
                K=K,
                Kp=Kp,
                Q=0,
                vH=vH
            )

            val = np.max(np.abs(P))

            print(
                f"K={K}  Kp={Kp}  Q=0  vH={vH}  max={val}"
            )
print("***********************************")
for K in [0,2]:
    for Kp in [0,2]:

        Qmax = min(K, Kp)

        for Q in range(-Qmax, Qmax+1):

            for vH in [0.0, 0.5, 2.0]:

                P = Phi_generalized(
                    x,
                    K=K,
                    Kp=Kp,
                    Q=Q,
                    vH=vH
                )

                print(
                    K, Kp, Q, vH,
                    np.max(np.abs(P))
                )

print("DEBUG")
Phi_generalized(
    np.array([0.0]),
    K=2,
    Kp=2,
    Q=2,
    vH=1.0
)
'''
print(
    "direct tests:",
    W3(1,0,1,-1,0,1),
    W3(1,0,1,0,0,0),
    W3(1,0,1,1,0,-1)
)

print(
    phi_transition(
        np.array([0.0]),
        +1,
        0,
        1.0
    )
)

print(
    phi_transition(
        np.array([0.0]),
        -1,
        0,
        1.0
    )
)'''
print("HEY")
for Q in [-2,-1,0,1,2]:
    val = np.max(
        np.abs(
            Phi_generalized(
                x,
                K=2,
                Kp=2,
                Q=Q,
                vH=1.0
            )
        )
    )

    print(Q, val)

print("Amplitude")
for Q in [-2,-1,0,1,2]:
    P = Phi_generalized(
        x,
        K=2,
        Kp=2,
        Q=Q,
        vH=1.0
    )

    print(
        Q,
        np.max(np.abs(P))
    )

for Q in [-2,-1,0,1,2]:
    y = Phi_generalized(
        np.array([0.0]),
        K=2,
        Kp=2,
        Q=Q,
        vH=1.0
    )
    print(Q, y[0])

plt.figure()
for Q in [-2,-1,0,1,2]:
    prof = Phi_generalized(x,2,2,Q,vH=2.0)
    plt.plot(x, prof, label = f"Phi22 Q={Q}")
    plt.title("Phi22")
plt.legend()
plt.savefig("Phi22_Qs_png", dpi = 300)
    

def stokes_profile_22_only(
x_grid,
Hu,
vH,
J_rad,
theta_B,
chi_B,
theta_obs,
chi_obs,
gamma_obs=np.pi/2):


    Jarr = Jrad_to_array(J_rad)

    rho20 = apply_hanle(
        Jarr,
        Hu,
        theta_B,
        chi_B
    )

    Qs = np.array([-2,-1,0,1,2])

    epsI = np.zeros_like(x_grid, dtype=complex)
    epsQ = np.zeros_like(x_grid, dtype=complex)
    epsU = np.zeros_like(x_grid, dtype=complex)

    for Q in Qs:

        rho = rho20[idx(-Q)]

        phase = (-1.0)**Q

        Phi22 = Phi_generalized(
            x_grid,
            K=2,
            Kp=2,
            Q=Q,
            vH=vH
        )

        epsI += (
            phase
            * Phi22
            * T(
                0,
                2,
                Q,
                theta_obs,
                chi_obs,
                gamma_obs
            )
            * rho
        )

        epsQ += (
            phase
            * Phi22
            * T(
                1,
                2,
                Q,
                theta_obs,
                chi_obs,
                gamma_obs
            )
            * rho
        )

        epsU += (
            phase
            * Phi22
            * T(
                2,
                2,
                Q,
                theta_obs,
                chi_obs,
                gamma_obs
            )
            * rho
        )

    return (
        np.real(epsI),
        np.real(epsQ),
        np.real(epsU)
    )

x = np.linspace(-5,5,401)
Jrad_0 = radiation_tensor(hR=0.073)
I22, Q22, U22 = stokes_profile_22_only(
x_grid=x,
Hu=1.0,
vH=0.002,
J_rad=Jrad_0,
theta_B=np.pi/2,
chi_B=0.0,
theta_obs=np.pi/2,
chi_obs=0.0
)

plt.figure(figsize=(8,5))
plt.plot(x, I22, label="I")
plt.plot(x, Q22, label="Q")
plt.plot(x, U22, label="U")
plt.legend()
plt.grid()
plt.savefig("222.png", dpi = 300)

def stokes_I_22_only(
x_grid,
Hu,
vH,
J_rad,
theta_B,
chi_B,
theta_obs,
chi_obs,
gamma_obs=np.pi/2):

    
    Jarr = Jrad_to_array(J_rad)

    rho20 = apply_hanle(
        Jarr,
        Hu,
        theta_B,
        chi_B
    )

    Qs = np.array([-2,-1,0,1,2])

    I = np.zeros_like(x_grid, dtype=float)

    for ix, x in enumerate(x_grid):

        epsI = 0.0j

        for Q in Qs:

            rho = rho20[idx(-Q)]

            Phi22 = Phi_generalized(
                x,
                K=2,
                Kp=2,
                Q=Q,
                vH=vH
            )

            epsI += (
                (-1.0)**Q
                * Phi22
                * T(
                    0,      # Stokes I
                    2,
                    Q,
                    theta_obs,
                    chi_obs,
                    gamma_obs
                )
                * rho
            )

        I[ix] = np.real(epsI)

    return I

x = np.linspace(-5,5,401)

I22 = stokes_I_22_only(
x,
Hu=1.0,
vH=0.002,
J_rad=Jrad_0,
theta_B=np.pi/2,
chi_B=0.0,
theta_obs=np.pi/2,
chi_obs=0.0
)

plt.figure()
plt.plot(x, I22)
plt.xlabel(r'$(\nu-\nu_0)/\Delta\nu_D$')
plt.ylabel(r'$I_{22}$')
plt.title("22 contribution only")
plt.grid(True)
plt.savefig("I22.png", dpi = 300)


x0 = np.array([0.0])   # line center

for Q in [-2,-1,0,1,2]:

    phi = Phi_generalized(
        x0,
        K=2,
        Kp=2,
        Q=Q,
        vH=2.0
    )[0]

    rho = rho20[idx(-Q)]

    tQ = T(
        1,      # Stokes Q
        2,
        Q,
        theta_obs,
        chi_obs,
        gamma_obs
    )

    contrib = ((-1)**Q) * phi * tQ * rho

    print(
        f"Q={Q:+d}",
        f"Phi={phi:+.6e}",
        f"rho={rho}",
        f"T={tQ}",
        f"contrib={contrib}"
    )

epsQ = 0.0 + 0.0j

for Q in [-2,-1,0,1,2]:

    term = (
        (-1)**Q
        * Phi_generalized(
            x0,
            2,
            2,
            Q,
            vH=2.0
        )[0]
        * T(
            1,
            2,
            Q,
            theta_obs,
            chi_obs,
            gamma_obs
        )
        * rho20[idx(-Q)]
    )

    print(f"Q={Q:+d}   term={term}")

    epsQ += term

print("epsQ =", epsQ)

for Q in [-2,-1,0,1,2]:
    print(
        Q,
        np.max(np.abs(
            Phi_generalized(x,2,2,Q,vH=2)
        ))
    )

for Q in [-2,-1,0,1,2]:
    print(
        Q,
        np.trapezoid(
            Phi_generalized(x,2,2,Q,vH=2),
            x
        )
    )


epsQ_Q0 = []
epsQ_Q2 = []
summ = []
x = np.linspace(-4,4,501)
for xx in x:

    term0 = (
        Phi_generalized(xx,2,2,0,vH)
        * T(1,2,0,theta_obs,chi_obs,gamma_obs)
        * rho20[idx(0)]
    )

    term2 = (
        Phi_generalized(xx,2,2,2,vH)
        * T(1,2,2,theta_obs,chi_obs,gamma_obs)
        * rho20[idx(-2)]
    )

    termm2 = (
        Phi_generalized(xx,2,2,-2,vH)
        * T(1,2,-2,theta_obs,chi_obs,gamma_obs)
        * rho20[idx(2)]
    )

    epsQ_Q0.append(np.real(term0))
    epsQ_Q2.append(np.real(term2 + termm2))
    summ.append(np.real(term0 + term2 + termm2))
plt.figure()
plt.plot(x, epsQ_Q0, label="Q=0 contribution")
plt.plot(x, epsQ_Q2, label="Q=±2 contribution")
plt.plot(x, summ, label = "sum")
plt.legend()
plt.savefig("Q_freq_dep.png", dpi = 300)

print("++++++++++++++++++++++++++++++++++++")
xgrid = np.linspace(-5,5,401)

I_prof = np.zeros_like(xgrid)
Q_prof = np.zeros_like(xgrid)
U_prof = np.zeros_like(xgrid)
V_prof = np.zeros_like(xgrid)

theta_B = np.pi/2
chi_B = 0.0
theta_obs = np.pi/2,
chi_obs = 0.0

for ix, x in enumerate(xgrid):
    Jarr = Jrad_to_array(Jrad_0)
    rho00 = Jrad_0[(0,0)]
    rho20 = apply_hanle(
        Jarr,
        Hu,
        theta_B,
        chi_B
    )
    epsI = 0.0+0j
    epsQ = 0.0+0j
    epsU = 0.0+0j
    epsV = 0.0+0j
    for K in [0, 1, 2]:
        for Kp in [0, 1, 2]:
            for Q in [-2,-1,0,1,2]:

                rho = rho20[idx(-Q)]

                phi22 = Phi_generalized(
                    np.array([x]),
                    K=K,
                    Kp=Kp,
                    Q=Q,
                    vH=0.002
                )[0]

                epsI += (
                    phi22
                    * (-1.0)**Q
                    * T(0,2,Q,theta_obs,chi_obs,np.pi/2)
                    * rho
                )

                epsQ += (
                    phi22
                    * (-1.0)**Q
                    * T(1,2,Q,theta_obs,chi_obs,np.pi/2)
                    * rho
                )

                epsU += (
                    phi22
                    * (-1.0)**Q
                    * T(2,2,Q,theta_obs,chi_obs,np.pi/2)
                    * rho
                )
                
                epsV +=(
                    phi22
                    * (-1.0)**Q
                    * T(3,2,Q,theta_obs,chi_obs,np.pi/2)
                    * rho
                )
            print("K = {}, Kp = {}, Q = {}".format(K, Kp, Q))
            print("Eps I", np.real(epsI))
            print("Eps Q", np.real(epsQ))
            print("Eps U", np.real(epsU))
            print(np.shape(np.real(epsV)))
            print("Eps V", np.real(epsV))
            '''
            # all Ks
            if K == 2:
                I_prof[ix] = (np.real(epsI[0]))
                Q_prof[ix] = (np.real(epsQ[0]))
                U_prof[ix] = (np.real(epsU[0]))
            elif K == 1:
                V_prof[ix] = (np.real(epsV[0]))
            else:
                I_prof[ix] = (np.real(epsI))
                Q_prof[ix] = (np.real(epsQ))
                U_prof[ix] = (np.real(epsU))
                V_prof[ix] = (np.real(epsV))
            '''
            I_prof[ix] = (np.real(epsI[0]))
            Q_prof[ix] = (np.real(epsQ[0]))
            U_prof[ix] = (np.real(epsU[0]))
            V_prof[ix] = np.real(epsV)

fig, ax = plt.subplots(2,2,figsize=(10,8))

ax[0,0].plot(xgrid,I_prof)
ax[0,0].set_title("I")

ax[0,1].plot(xgrid,Q_prof)
ax[0,1].set_title("Q")

ax[1,0].plot(xgrid,U_prof)
ax[1,0].set_title("U")

ax[1,1].plot(xgrid,V_prof)
ax[1,1].set_title("V")

plt.tight_layout()
plt.savefig("Stokes_try.png", dpi = 300)

def emissivity_profile(
        x,
        rho20,
        theta_obs,
        chi_obs,
        vH):

    epsI = 0.0+0j
    epsQ = 0.0+0j
    epsU = 0.0+0j

    for Q in [-2,-1,0,1,2]:

        Phi = Phi_generalized(
            np.array([x]),
            K=2,
            Kp=2,
            Q=Q,
            vH=vH
        )[0]

        rho = rho20[idx(-Q)]

        phase = (-1)**Q

        epsI += (
            phase
            * T(0,2,Q,
                theta_obs,
                chi_obs,
                np.pi/2)
            * rho
            * Phi
        )

        epsQ += (
            phase
            * T(1,2,Q,
                theta_obs,
                chi_obs,
                np.pi/2)
            * rho
            * Phi
        )

        epsU += (
            phase
            * T(2,2,Q,
                theta_obs,
                chi_obs,
                np.pi/2)
            * rho
            * Phi
        )

    return (
        np.real(epsI),
        np.real(epsQ),
        np.real(epsU)
    )
xgrid = np.linspace(-5,5,401)

Iprof = np.zeros_like(xgrid)
Qprof = np.zeros_like(xgrid)
Uprof = np.zeros_like(xgrid)

rho20 = apply_hanle(
    Jarr,
    Hu,
    theta_B,
    chi_B
)

for i,x in enumerate(xgrid):

    I,Q,U = emissivity_profile(
        x,
        rho20,
        theta_obs,
        chi_obs,
        vH
    )

    Iprof[i] = I[0]
    Qprof[i] = Q[0]
    Uprof[i] = U[0]

fig,ax = plt.subplots(2,2,figsize=(10,8))

ax[0,0].plot(xgrid,Iprof)
ax[0,0].set_title("I")

ax[0,1].plot(xgrid,Qprof)
ax[0,1].set_title("Q")

ax[1,0].plot(xgrid,Uprof)
ax[1,0].set_title("U")

ax[1,1].axis("off")

plt.tight_layout()
plt.savefig("Stokes_try22.png", dpi = 300)

def stokes_profiles_LL04(
        xgrid,
        Hu,
        Jrad,
        theta_B,
        chi_B,
        theta_obs,
        chi_obs,
        vH,
        gamma_obs=np.pi/2):

    # --------------------------------------
    # Radiation tensors
    # --------------------------------------

    J00 = Jrad[(0,0)]

    Jvert = Jrad_to_array(Jrad)

    # full Hanle solution:
    # rho_Q = [D H D† J]_Q
    rho = apply_hanle(
        Jvert,
        Hu,
        theta_B,
        chi_B
    )
    print("rho components")
    for Q in [-2,-1,0,1,2]:
        print(Q, rho[idx(Q)])
    # --------------------------------------
    # output arrays
    # --------------------------------------

    Iprof = np.zeros_like(xgrid)
    Qprof = np.zeros_like(xgrid)
    Uprof = np.zeros_like(xgrid)

    # --------------------------------------
    # frequency loop
    # --------------------------------------

    for ix, x in enumerate(xgrid):

        epsI = 0.0 + 0.0j
        epsQ = 0.0 + 0.0j
        epsU = 0.0 + 0.0j

        # =====================================================
        # (K,K') = (0,0)
        # =====================================================

        Phi00 = Phi_generalized(
            np.array([x]),
            K=0,
            Kp=0,
            Q=0,
            vH=vH
        )[0]

        epsI += (
            Phi00
            * T(
                0,      # Stokes I
                0,
                0,
                theta_obs,
                chi_obs,
                gamma_obs
            )
            * J00
        )

        # =====================================================
        # (K,K') = (0,2)
        # =====================================================

        Phi02 = Phi_generalized(
            np.array([x]),
            K=0,
            Kp=2,
            Q=0,
            vH=vH
        )[0]

        epsI += (
            Phi02
            * T(
                0,
                2,
                0,
                theta_obs,
                chi_obs,
                gamma_obs
            )
            * J00
        )

        epsQ += (
            Phi02
            * T(
                1,
                2,
                0,
                theta_obs,
                chi_obs,
                gamma_obs
            )
            * J00
        )

        epsU += (
            Phi02
            * T(
                2,
                2,
                0,
                theta_obs,
                chi_obs,
                gamma_obs
            )
            * J00
        )

        # =====================================================
        # K = 2 terms
        # =====================================================
        # =====================================================
        # (2,0)
        # =====================================================

        Phi20 = Phi_generalized(
            np.array([x]),
            K=2,
            Kp=0,
            Q=0,
            vH=vH
        )[0]

        rho20 = rho[idx(0)]

        epsI += (
            Phi20
            * T(
                0,
                0,
                0,
                theta_obs,
                chi_obs,
                gamma_obs
            )
            * rho20
        )

        for Q in [-2,-1,0,1,2]:

            phase = (-1)**Q

            rhoQ = rho[idx(-Q)]

            # -------------------------------
            # (2,2)
            # -------------------------------

            Phi22 = Phi_generalized(
                np.array([x]),
                K=2,
                Kp=2,
                Q=Q,
                vH=vH
            )[0]

            epsI += (
                phase
                * Phi22
                * T(
                    0,
                    2,
                    Q,
                    theta_obs,
                    chi_obs,
                    gamma_obs
                )
                * rhoQ
            )

            epsQ += (
                phase
                * Phi22
                * T(
                    1,
                    2,
                    Q,
                    theta_obs,
                    chi_obs,
                    gamma_obs
                )
                * rhoQ
            )

            epsU += (
                phase
                * Phi22
                * T(
                    2,
                    2,
                    Q,
                    theta_obs,
                    chi_obs,
                    gamma_obs
                )
                * rhoQ
            )

        Iprof[ix] = np.real(epsI)
        Qprof[ix] = np.real(epsQ)
        Uprof[ix] = np.real(epsU)

    return Iprof, Qprof, Uprof

xgrid = np.linspace(-5,5,401)

I,Q,U = stokes_profiles_LL04(
    xgrid,
    Hu=1.0,
    Jrad=Jrad_0,
    theta_B=np.pi/2,
    chi_B=0.0,
    theta_obs=np.pi/2,
    chi_obs=0.0,
    vH=1.0
)

fig,ax = plt.subplots(2,2,figsize=(10,8))

ax[0,0].plot(xgrid,I)
ax[0,0].set_title("I")

ax[0,1].plot(xgrid,Q)
ax[0,1].set_title("Q")

ax[1,0].plot(xgrid,U)
ax[1,0].set_title("U")

ax[1,1].axis("off")

plt.tight_layout()
plt.savefig("Stokes_try_noV.png", dpi = 300)