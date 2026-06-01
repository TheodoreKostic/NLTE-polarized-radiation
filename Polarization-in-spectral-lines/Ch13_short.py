import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from functions_prt import short_characteristics

# Allen (1973) limb darkening coefficients for the Sun
# for biquadratic law: I(μ)/I(1) = 1 - u1*(1-μ) - u2*(1-μ)^2

u1 = 0.95
u2 = -0.20

# Define functions for anisotropy and geometry based on
# Polarization in Spectral Lines, Landi Degl'Innocenti & Landolfi (2004), Chapter 13

def anisotropy(hR):

    S = 1.0/(1.0 + hR)
    C = np.sqrt(1.0 - S**2)

    a0 = 1 - C

    a1 = (
    C
    - 0.5
    - 0.5*(C**2/S)*np.log((1+S)/C)
    )

    a2 = ((C+2)*(C-1))/ (3*(C+1))

    b0 = (1 - C**3)/3

    b1 = (
        (8*C**3 - 3*C**2 - 2)/24
        - (C**4)/(8*S)*np.log((1+S)/C)
    )

    b2 = (
        (C-1)*(3*C**3 + 6*C**2 + 4*C + 2)
        /(15*(C+1))
    )

    A = a0 + a1*u1 + a2*u2
    B = b0 + b1*u1 + b2*u2

    return (3*B - A)/(2*A)

def pQ(hR, delta_deg):

    d = np.radians(delta_deg)

    w = anisotropy(hR)

    return (
        3*np.cos(d)**2 * w /
        (4 + (3*np.sin(d)**2 - 1)*w)
    )


hp = np.linspace(0,0.2,400)

plt.figure(figsize=(8,6))
for d in [0,10,20,30]:

    hR = (1 + hp)/np.cos(np.radians(d)) - 1

    plt.plot(hp, pQ(hR,d), label=f"$\\delta = {d}°$")

plt.xlabel(r"$h'/R_\odot$")
plt.ylabel(r"$p_Q$")
plt.legend()
plt.grid()
plt.savefig("Fig13.2.png")

for h in [1e-4,0.05,0.1,0.2]:
    print(anisotropy(h))

for h in [1e-4,0.05,0.1,0.2]:

    S = 1/(1+h)
    C = np.sqrt(1-S**2)

    a0 = 1-C

    a1 = (
    C
    - 0.5
    - 0.5*(C**2/S)*np.log((1+S)/C)
    )

    a2 = ((C+2)*(C-1))/(3*(C+1))

    b0 = (1-C**3)/3

    b1 = (
        (8*C**3 - 3*C**2 - 2)/24
        - (C**4)/(8*S)*np.log((1+S)/C)
    )

    b2 = (
        (C-1)*(3*C**3 + 6*C**2 + 4*C + 2)
        /(15*(C+1))
    )

    A = a0 + a1*u1 + a2*u2
    B = b0 + b1*u1 + b2*u2

    print(h)
    print("A =",A)
    print("B =",B)
    print("3B/A =",3*B/A)
    print()

    c0 = 3*b0 - a0
    c1 = 3*b1 - a1
    c2 = 3*b2 - a2

    w1 = (3*B - A)/(2*A)
    w2 = 0.5*(c0 + c1*u1 + c2*u2)/A

    print("w1 =", w1)
    print("w2 =", w2)
    print("difference =", w1-w2)
    print("\n")

    