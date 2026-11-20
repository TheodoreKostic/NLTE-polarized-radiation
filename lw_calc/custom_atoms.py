"""
Custom atomic models for sr_synthesis project.

This module contains custom atomic model definitions that are not part
of the standard lightweaver.rh_atoms library.
"""

''' 
IMPORTANT. These are collisional rates used by M. Bergemann for Sr I 460.7 nm line

CE 
3472.5 4575.3 4954.1 6423.6 9495.7
4.92E-06 1.46E-06 1.25E-06 8.71E-07 3.00E-07
CH
1.27E-16    4.52E-16    6.45E-16    2.08E-15    1.12E-14
CI
2.68E-14    3.46E-13    6.45E-13    3.68E-12    2.59E-11
'''

from fractions import Fraction
from lightweaver.atomic_model import AtomicModel, AtomicLevel, AtomicLine, AtomicContinuum, HydrogenicContinuum, ExplicitContinuum
from lightweaver.atomic_model import VoigtLine, LinearCoreExpWings, LineType, LineQuadrature
from lightweaver import Element
from lightweaver.collisional_rates import CE, CI, CH, CollisionalRates
from lightweaver.broadening import LineBroadening, RadiativeBroadening, VdwUnsold, MultiplicativeStarkBroadening


def sr_tanausu():
    """
    Custom Sr I atomic model based on SrI.atom/StrontiumI.atom (HanleRT, Tanausu)
    NIST has slighly different energies!

    Contains 16 levels (15 Sr I + 1 Sr II ground state) + 12 lines 

    Returns
    -------
    AtomicModel
        Sr I atomic model
    """
    return AtomicModel(element = Element(Z=38),
    levels = [
        AtomicLevel(E=     0.000, g=1, label="SR I 5S2 1SE 0",  stage=0, J=Fraction(0, 1), L=0, S=Fraction(0, 1)), #0
        AtomicLevel(E= 14317.507, g=1, label="SR I 5S5P 3P0 0", stage=0, J=Fraction(0, 1), L=1, S=Fraction(1, 1)), #1
        AtomicLevel(E= 14504.334, g=3, label="SR I 5S5P 3P0 1", stage=0, J=Fraction(1, 1), L=1, S=Fraction(1, 1)), #2
        AtomicLevel(E= 14898.545, g=5, label="SR I 5S5P 3P0 2", stage=0, J=Fraction(2, 1), L=1, S=Fraction(1, 1)), #3
        AtomicLevel(E= 18159.039, g=3, label="SR I 5S4D 3DE 1", stage=0, J=Fraction(1, 1), L=2, S=Fraction(1, 1)), #4
        AtomicLevel(E= 18218.783, g=5, label="SR I 5S4D 3DE 2", stage=0, J=Fraction(2, 1), L=2, S=Fraction(1, 1)), #5
        AtomicLevel(E= 18319.262, g=7, label="SR I 5S4D 3DE 3", stage=0, J=Fraction(3, 1), L=2, S=Fraction(1, 1)), #6
        AtomicLevel(E= 20149.686, g=5, label="SR I 5S4D 1DE 2", stage=0, J=Fraction(2, 1), L=2, S=Fraction(0, 1)), #7
        AtomicLevel(E= 21698.452, g=3, label="SR I 5S5P 1P0 1", stage=0, J=Fraction(1, 1), L=1, S=Fraction(0, 1)), #8
        AtomicLevel(E= 30591.824, g=3, label="SR I 5S6S 1SE 0", stage=0, J=Fraction(0, 1), L=0, S=Fraction(0, 1)), #9
        AtomicLevel(E= 34098.402, g=1, label="SR I 5S6P 1P0 1", stage=0, J=Fraction(1, 1), L=1, S=Fraction(0, 1)), #10
        AtomicLevel(E= 34727.445, g=5, label="SR I 5S5D 1DE 2", stage=0, J=Fraction(2, 1), L=2, S=Fraction(0, 1)), #11
        AtomicLevel(E= 38906.859, g=3, label="SR I 5S7P 1P0 1", stage=0, J=Fraction(1, 1), L=1, S=Fraction(0, 1)), #12 not in original lightweaver RH atoms
        AtomicLevel(E= 39539.012, g=7, label="SR I 5S4F 1F0 3", stage=0, J=Fraction(3, 1), L=3, S=Fraction(0, 1)), #13 not in original lightweaver RH atoms
        AtomicLevel(E= 39733.066, g=5, label="SR I 5S6D 1DE 2", stage=0, J=Fraction(2, 1), L=2, S=Fraction(0, 1)), #14 not in original lightweaver RH atoms
        AtomicLevel(E= 45932.203, g=2, label="SR II 4P6S 2SE 1", stage=1, J=Fraction(1, 2), L=0, S=Fraction(1, 2)),#15 
    ],
    lines = [
        VoigtLine(j=2,  i=0, f=1.000e-03, type=LineType.CRD, quadrature=LinearCoreExpWings(qCore=3, qWing=100, Nlambda=101), broadening=LineBroadening(natural=[RadiativeBroadening(gamma=1.40e+05)], elastic=[VdwUnsold(vals=[1.5, 1.5]), MultiplicativeStarkBroadening(coeff=0)])),# NIST fik confirmed 2026-05-28
        VoigtLine(j=8,  i=0, f=1.920e+00, type=LineType.CRD, quadrature=LinearCoreExpWings(qCore=3, qWing=100, Nlambda=101), broadening=LineBroadening(natural=[RadiativeBroadening(gamma=6.03e+08)], elastic=[VdwUnsold(vals=[1.5, 1.5]), MultiplicativeStarkBroadening(coeff=0)])),# NIST fik confirmed 2026-05-28
        VoigtLine(j=10, i=0, f=7.300e-03, type=LineType.CRD, quadrature=LinearCoreExpWings(qCore=3, qWing=100, Nlambda=101), broadening=LineBroadening(natural=[RadiativeBroadening(gamma=1.12e+07)], elastic=[VdwUnsold(vals=[1.5, 1.5]), MultiplicativeStarkBroadening(coeff=0)])),# NIST fik confirmed 2026-05-28
        VoigtLine(j=12, i=0, f=1.600e-02, type=LineType.CRD, quadrature=LinearCoreExpWings(qCore=3, qWing=100, Nlambda=101), broadening=LineBroadening(natural=[RadiativeBroadening(gamma=1.74e+07)], elastic=[VdwUnsold(vals=[1.5, 1.5]), MultiplicativeStarkBroadening(coeff=0)])),# NIST fik confirmed 2026-05-28
        VoigtLine(j=10, i=7, f=4.300e-02, type=LineType.CRD, quadrature=LinearCoreExpWings(qCore=1, qWing=100, Nlambda=101), broadening=LineBroadening(natural=[RadiativeBroadening(gamma=1.12e+07)], elastic=[VdwUnsold(vals=[1.5, 1.5]), MultiplicativeStarkBroadening(coeff=0)])),# No fik data in NIST
        VoigtLine(j=13, i=7, f=1.500e-01, type=LineType.CRD, quadrature=LinearCoreExpWings(qCore=3, qWing=100, Nlambda=101), broadening=LineBroadening(natural=[RadiativeBroadening(gamma=4.30e+07)], elastic=[VdwUnsold(vals=[1.5, 1.5]), MultiplicativeStarkBroadening(coeff=0)])),# No fik data in NIST
        VoigtLine(j=9,  i=8, f=5.556e-03, type=LineType.CRD, quadrature=LinearCoreExpWings(qCore=3, qWing=100, Nlambda=101), broadening=LineBroadening(natural=[RadiativeBroadening(gamma=2.93e+05)], elastic=[VdwUnsold(vals=[1.5, 1.5]), MultiplicativeStarkBroadening(coeff=0)])),# No fik data in NIST
        VoigtLine(j=11, i=8, f=9.700e-02, type=LineType.CRD, quadrature=LinearCoreExpWings(qCore=3, qWing=100, Nlambda=101), broadening=LineBroadening(natural=[RadiativeBroadening(gamma=1.10e+07)], elastic=[VdwUnsold(vals=[1.5, 1.5]), MultiplicativeStarkBroadening(coeff=0)])),# NIST fik confirmed 2026-05-28
        VoigtLine(j=14, i=8, f=4.272e-02, type=LineType.CRD, quadrature=LinearCoreExpWings(qCore=3, qWing=100, Nlambda=101), broadening=LineBroadening(natural=[RadiativeBroadening(gamma= 1.12e+07)],elastic=[VdwUnsold(vals=[1.5, 1.5]), MultiplicativeStarkBroadening(coeff=0)])),# No fik data in NIST
        VoigtLine(j=12, i=9, f=2.641e-02, type=LineType.CRD, quadrature=LinearCoreExpWings(qCore=3, qWing=100, Nlambda=101), broadening=LineBroadening(natural=[RadiativeBroadening(gamma=1.74e+07)], elastic=[VdwUnsold(vals=[1.5, 1.5]), MultiplicativeStarkBroadening(coeff=0)])),# No fik data in NIST
        VoigtLine(j=14, i=10, f=9.270e-02, type=LineType.CRD, quadrature=LinearCoreExpWings(qCore=3, qWing=100, Nlambda=101), broadening=LineBroadening(natural=[RadiativeBroadening(gamma=1.12e+07)],elastic=[VdwUnsold(vals=[1.5, 1.5]), MultiplicativeStarkBroadening(coeff=0)])),# No fik data in NIST
        VoigtLine(j=13, i=11, f=3.500e-01, type=LineType.CRD, quadrature=LinearCoreExpWings(qCore=3, qWing=100, Nlambda=101), broadening=LineBroadening(natural=[RadiativeBroadening(gamma=4.30e+07)],elastic=[VdwUnsold(vals=[1.5, 1.5]), MultiplicativeStarkBroadening(coeff=0)])),# NIST fik confirmed 2026-05-28
    ],
    continua = [ # Photoionization Cross Sections
      ExplicitContinuum(j=15, i=0, wavelengthGrid=[74.815002, 81.502002, 89.503003, 99.246002, 111.36801, 126.86400, 147.37000, 160.32700, 175.78199, 182.83199, 190.47100, 198.77600, 207.83899, 217.76699], alphaGrid=[8.6310000e-23, 1.0180000e-22, 1.2120000e-22, 1.4650000e-22, 1.7979999e-22, 2.2469999e-22, 2.8540000e-22, 3.2320000e-22, 3.6800000e-22, 3.8720002e-22, 4.0819998e-22, 4.2969998e-22, 4.5260002e-22, 4.7520000e-22]),
      ExplicitContinuum(j=15, i=1, wavelengthGrid=[83.791998, 92.271997, 102.66100, 115.68700, 132.49900, 155.02800, 186.78900, 208.10601, 234.91499, 247.67800, 261.90701, 277.87100, 295.90801, 316.44900], alphaGrid=[1.0850000e-22, 1.3090000e-22, 1.6100000e-22, 2.0500001e-22, 2.6860000e-22, 3.6960001e-22, 5.4140001e-22, 6.7560001e-22, 8.6359997e-22, 9.6099999e-22, 1.0730000e-21, 1.2060000e-21, 1.3620000e-21, 1.5489999e-21]),
      ExplicitContinuum(j=15, i=2, wavelengthGrid=[83.921997, 92.429999, 102.85699, 115.93600, 132.82600, 155.47600, 187.43800, 208.91201, 235.94299, 248.82100, 263.18601, 279.31201, 297.54199, 318.31799], alphaGrid=[1.0670000e-22, 1.2890000e-22, 1.5870000e-22, 2.0239999e-22, 2.6570001e-22, 3.6610001e-22, 5.3709999e-22, 6.7109999e-22, 8.5859999e-22, 9.5610002e-22, 1.0680000e-21, 1.2010000e-21, 1.3570000e-21, 1.5439999e-21]),
      ExplicitContinuum(j=15, i=3, wavelengthGrid=[84.201001, 92.769000, 103.27700, 116.47000, 133.52700, 156.43700, 188.83700, 210.65200, 238.16499, 251.29299, 265.95300, 282.43000, 301.08301, 322.37400], alphaGrid=[1.0280000e-22, 1.2460000e-22, 1.5380000e-22, 1.9680000e-22, 2.5920000e-22, 3.5849999e-22, 5.2829999e-22, 6.6190000e-22, 8.4920000e-22, 9.4690003e-22, 1.0590000e-21, 1.1920000e-21, 1.3500000e-21, 1.5380001e-21]),
      ExplicitContinuum(j=15, i=4, wavelengthGrid=[86.577002, 95.660999, 106.87400, 121.06500, 139.60200, 164.84200, 201.22200, 226.18101, 258.20801, 273.71101, 291.19500, 311.06399, 333.84399, 360.22300], alphaGrid=[3.8359999e-23, 1.1810000e-23, 1.3750000e-23, 8.0150001e-23, 2.2180000e-22, 5.8400002e-22, 1.2030000e-21, 1.6770000e-21, 2.3530000e-21, 2.7090000e-21, 3.0760001e-21, 3.4780000e-21, 4.0370001e-21, 4.7050000e-21]),
      ExplicitContinuum(j=15, i=5, wavelengthGrid=[86.626001, 95.720001, 106.94800, 121.16000, 139.72800, 165.01700, 201.48400, 226.51201, 258.63899, 274.19600, 291.74299, 311.68999, 334.56499, 361.06299], alphaGrid=[3.9140000e-23, 1.2190000e-23, 1.3080000e-23, 7.7759998e-23, 2.1670000e-22, 5.7430000e-22, 1.1880000e-21, 1.6590001e-21, 2.3339999e-21, 2.6890001e-21, 3.0559999e-21, 3.4580001e-21, 4.0180000e-21, 4.6869999e-21]),
      ExplicitContinuum(j=15, i=6, wavelengthGrid=[86.697998, 95.809003, 107.05900, 121.30200, 139.91700, 165.28199, 201.87800, 227.01001, 259.28899, 274.92500, 292.56899, 312.63301, 335.65200, 362.32900], alphaGrid=[3.6759999e-23, 1.2889999e-23, 1.8300000e-23, 8.8580001e-23, 2.3109999e-22, 5.9340002e-22, 1.2110000e-21, 1.6850000e-21, 2.3620000e-21, 2.7180000e-21, 3.0870000e-21, 3.4919999e-21, 4.0560001e-21, 4.7330001e-21]),
      ExplicitContinuum(j=15, i=7, wavelengthGrid=[88.096997, 97.520001, 109.20000, 124.05800, 143.59700, 170.44100, 209.62800, 236.85601, 272.21499, 289.50100, 309.13301, 331.62000, 357.63501, 388.08000], alphaGrid=[4.4799999e-22, 3.6169998e-22, 2.4519999e-22, 1.2650000e-22, 4.2059999e-23, 1.1590000e-23, 1.4570000e-22, 3.2840001e-22, 6.7010001e-22, 8.7950003e-22, 1.1110000e-21, 1.3750000e-21, 1.7740000e-21, 2.2620000e-21]),
      ExplicitContinuum(j=15, i=8, wavelengthGrid=[89.315997, 99.015997, 111.07900, 126.48900, 146.86400, 175.06200, 216.66299, 245.87700, 284.19800, 303.09299, 324.67900, 349.57700, 378.60901, 412.90098], alphaGrid=[5.1239998e-23, 6.5050000e-23, 8.4149999e-23, 1.1260000e-22, 1.5650000e-22, 2.2900001e-22, 3.5990000e-22, 4.6749998e-22, 6.2259999e-22, 7.0579996e-22, 8.0230001e-22, 9.1710003e-22, 1.0570000e-21, 1.2220000e-21]),
      ExplicitContinuum(j=15, i=9, wavelengthGrid=[97.028003, 108.58300, 123.26300, 142.53199, 168.94200, 207.36599, 268.41201, 314.74099, 380.39900, 415.02998, 456.59902, 507.42100, 570.97500, 652.72700], alphaGrid=[3.7399999e-23, 4.7480002e-23, 6.1960000e-23, 8.3920001e-23, 1.1970000e-22, 1.8189999e-22, 3.0400000e-22, 4.1439999e-22, 5.9520000e-22, 7.0179997e-22, 8.3870001e-22, 1.0170000e-21, 1.2590000e-21, 1.5920001e-21]),
      ExplicitContinuum(j=15, i=10, wavelengthGrid=[100.44800, 112.88500, 128.83600, 150.03700, 179.59000, 223.64099, 296.32500, 353.82300, 439.00400, 485.78501, 543.72500, 617.35698, 714.05601, 846.67305], alphaGrid=[2.2110000e-23, 2.8389999e-23, 3.7020000e-23, 5.1890001e-23, 7.5939999e-23, 1.1780000e-22, 1.9050001e-22, 2.5150001e-22, 3.5499999e-22, 4.1019998e-22, 4.8130000e-22, 5.6529999e-22, 6.6460000e-22, 7.6090004e-22]),
      ExplicitContinuum(j=15, i=11, wavelengthGrid=[101.08700, 113.69200, 129.88900, 151.46700, 181.64301, 226.83301, 301.95601, 361.88000, 451.47700, 501.10298, 562.98701, 642.31001, 747.65000, 894.32100], alphaGrid=[2.0840000e-24, 3.0350000e-24, 4.6190001e-24, 7.4659997e-24, 1.3580000e-23, 2.8549999e-23, 7.4229999e-23, 1.3500000e-22, 2.7889999e-22, 3.9240001e-22, 5.7160000e-22, 8.7420004e-22, 1.4160000e-21, 2.4700000e-21]),
      ExplicitContinuum(j=15, i=12, wavelengthGrid=[105.54600, 119.36300, 137.34399, 161.70200, 196.56300, 250.58701, 345.56101, 426.35801, 556.46602, 633.83599, 736.19302, 877.97803, 1087.4000, 1428.0300], alphaGrid=[7.9529997e-24, 1.0510000e-23, 1.4050000e-23, 2.0310000e-23, 3.0649999e-23, 4.9200000e-23, 8.2780002e-23, 1.0040000e-22, 1.4720000e-22, 1.7190000e-22, 1.9690001e-22, 2.2839999e-22, 3.5559998e-22, 5.2840000e-22]),
      HydrogenicContinuum(j=15, i=13, NlambdaGen=14, alpha0=3.264e-21, minWavelength=100.00),
      ExplicitContinuum(j=15, i=14, wavelengthGrid=[106.47000, 120.54700, 138.91400, 163.88300, 199.79500, 255.86299, 355.67500, 441.86001, 583.17100, 668.71401, 783.66899, 946.35000, 1194.2700, 1618.1800], alphaGrid=[2.4030001e-25, 3.7210001e-25, 7.0580002e-25, 1.1890000e-24, 2.6000001e-24, 6.8930001e-24, 2.4339999e-23, 5.3980002e-23, 1.4370000e-22, 2.2900001e-22, 3.8859998e-22, 7.1559999e-22, 1.4760001e-21, 3.6119999e-21]),
    ],
    collisions = [
      CE(j=1, i=0, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[2.877e-15, 1.726e-15, 1.233e-15, 8.630e-16, 4.315e-16, 2.877e-16, 8.630e-17]), # Belly and Van Regermorter neutral
      CE(j=2, i=0, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[2.877e-15, 1.726e-15, 1.233e-15, 8.630e-16, 4.315e-16, 2.877e-16, 8.630e-17]), # Belly and Van Regermorter neutral
      CE(j=3, i=0, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[2.877e-15, 1.726e-15, 1.233e-15, 8.630e-16, 4.315e-16, 2.877e-16, 8.630e-17]), # Belly and Van Regermorter neutral
      CE(j=4, i=0, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[2.877e-15, 1.726e-15, 1.233e-15, 8.630e-16, 4.315e-16, 2.877e-16, 8.630e-17]), # Belly and Van Regermorter neutral
      CE(j=5, i=0, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[2.877e-15, 1.726e-15, 1.233e-15, 8.630e-16, 4.315e-16, 2.877e-16, 8.630e-17]), # Belly and Van Regermorter neutral
      CE(j=6, i=0, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[2.877e-15, 1.726e-15, 1.233e-15, 8.630e-16, 4.315e-16, 2.877e-16, 8.630e-17]), # Belly and Van Regermorter neutral
      CE(j=7, i=0, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[2.877e-15, 1.726e-15, 1.233e-15, 8.630e-16, 4.315e-16, 2.877e-16, 8.630e-17]), # Belly and Van Regermorter neutral
      #CE(j=8, i=0, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[8.291e-15, 8.180e-15, 6.720e-15, 6.090e-15, 4.817e-15, 3.991e-15, 3.192e-15]), # Van Regermorter neutral
      CE(j=8, i=0, temperature=[3472.5 , 4575.3,  4954.1,  6423.6, 9495.7], rates=[4.92E-12, 1.46E-12, 1.25E-12, 8.71E-13, 3.00E-13]), # New Values from Maria Bergemann
      CH(j=0, i=8, temperature=[3472.5 , 4575.3,  4954.1,  6423.6, 9495.7], rates=[ 1.27E-22, 4.52E-22, 6.45E-22,  2.08E-21, 1.12E-20], ),
      CE(j=9, i=0, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[2.877e-15, 1.726e-15, 1.233e-15, 8.630e-16, 4.315e-16, 2.877e-16, 8.630e-17]), # Belly and Van Regermorter neutral
      CE(j=10, i=0, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[1.600e-17, 1.384e-17, 1.323e-17, 1.101e-17, 8.499e-18, 7.499e-18, 5.663e-18]), # Van Regermorter neutral
      CE(j=11, i=0, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[2.877e-15, 1.726e-15, 1.233e-15, 8.630e-16, 4.315e-16, 2.877e-16, 8.630e-17]), # Belly and Van Regermorter neutral
      CE(j=12, i=0, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[2.877e-17, 2.229e-17, 2.314e-17, 2.004e-17, 1.520e-17, 1.280e-17, 1.017e-17]), # Van Regermorter neutral
      CE(j=13, i=0, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[2.877e-15, 1.726e-15, 1.233e-15, 8.630e-16, 4.315e-16, 2.877e-16, 8.630e-17]), # Belly and Van Regermorter neutral
      CE(j=14, i=0, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[2.877e-15, 1.726e-15, 1.233e-15, 8.630e-16, 4.315e-16, 2.877e-16, 8.630e-17]), # Belly and Van Regermorter neutral
      CE(j=2, i=1, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[2.877e-15, 1.726e-15, 1.233e-15, 8.630e-16, 4.315e-16, 2.877e-16, 8.630e-17]), # Belly and Van Regermorter neutral
      CE(j=3, i=1, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[2.877e-15, 1.726e-15, 1.233e-15, 8.630e-16, 4.315e-16, 2.877e-16, 8.630e-17]), # Belly and Van Regermorter neutral
      CE(j=4, i=1, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[2.877e-15, 1.726e-15, 1.233e-15, 8.630e-16, 4.315e-16, 2.877e-16, 8.630e-17]), # Belly and Van Regermorter neutral
      CE(j=5, i=1, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[2.877e-15, 1.726e-15, 1.233e-15, 8.630e-16, 4.315e-16, 2.877e-16, 8.630e-17]), # Belly and Van Regermorter neutral
      CE(j=6, i=1, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[2.877e-15, 1.726e-15, 1.233e-15, 8.630e-16, 4.315e-16, 2.877e-16, 8.630e-17]), # Belly and Van Regermorter neutral
      CE(j=7, i=1, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[2.877e-15, 1.726e-15, 1.233e-15, 8.630e-16, 4.315e-16, 2.877e-16, 8.630e-17]), # Belly and Van Regermorter neutral
      CE(j=8, i=1, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[2.877e-15, 1.726e-15, 1.233e-15, 8.630e-16, 4.315e-16, 2.877e-16, 8.630e-17]), # Belly and Van Regermorter neutral
      CE(j=9, i=1, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[2.877e-15, 1.726e-15, 1.233e-15, 8.630e-16, 4.315e-16, 2.877e-16, 8.630e-17]), # Belly and Van Regermorter neutral
      CE(j=10, i=1, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[2.877e-15, 1.726e-15, 1.233e-15, 8.630e-16, 4.315e-16, 2.877e-16, 8.630e-17]), # Belly and Van Regermorter neutral
      CE(j=11, i=1, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[2.877e-15, 1.726e-15, 1.233e-15, 8.630e-16, 4.315e-16, 2.877e-16, 8.630e-17]), # Belly and Van Regermorter neutral
      CE(j=12, i=1, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[2.877e-15, 1.726e-15, 1.233e-15, 8.630e-16, 4.315e-16, 2.877e-16, 8.630e-17]), # Belly and Van Regermorter neutral
      CE(j=13, i=1, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[2.877e-15, 1.726e-15, 1.233e-15, 8.630e-16, 4.315e-16, 2.877e-16, 8.630e-17]), # Belly and Van Regermorter neutral
      CE(j=14, i=1, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[2.877e-15, 1.726e-15, 1.233e-15, 8.630e-16, 4.315e-16, 2.877e-16, 8.630e-17]), # Belly and Van Regermorter neutral   
        CE(j=3, i=2, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[9.589e-16, 5.753e-16, 4.110e-16, 2.877e-16, 1.438e-16, 9.589e-17, 2.877e-17]), # Belly and Van Regermorter neutral
        CE(j=4, i=2, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[9.589e-16, 5.753e-16, 4.110e-16, 2.877e-16, 1.438e-16, 9.589e-17, 2.877e-17]), # Belly and Van Regermorter neutral
        CE(j=5, i=2, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[9.589e-16, 5.753e-16, 4.110e-16, 2.877e-16, 1.438e-16, 9.589e-17, 2.877e-17]), # Belly and Van Regermorter neutral
        CE(j=6, i=2, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[9.589e-16, 5.753e-16, 4.110e-16, 2.877e-16, 1.438e-16, 9.589e-17, 2.877e-17]), # Belly and Van Regermorter neutral
        CE(j=7, i=2, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[9.589e-16, 5.753e-16, 4.110e-16, 2.877e-16, 1.438e-16, 9.589e-17, 2.877e-17]), # Belly and Van Regermorter neutral
        CE(j=8, i=2, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[9.589e-16, 5.753e-16, 4.110e-16, 2.877e-16, 1.438e-16, 9.589e-17, 2.877e-17]), # Belly and Van Regermorter neutral
        CE(j=9, i=2, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[9.589e-16, 5.753e-16, 4.110e-16, 2.877e-16, 1.438e-16, 9.589e-17, 2.877e-17]), # Belly and Van Regermorter neutral
        CE(j=10, i=2, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[9.589e-16, 5.753e-16, 4.110e-16, 2.877e-16, 1.438e-16, 9.589e-17, 2.877e-17]), # Belly and Van Regermorter neutral
        CE(j=11, i=2, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[9.589e-16, 5.753e-16, 4.110e-16, 2.877e-16, 1.438e-16, 9.589e-17, 2.877e-17]), # Belly and Van Regermorter neutral
        CE(j=12, i=2, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[9.589e-16, 5.753e-16, 4.110e-16, 2.877e-16, 1.438e-16, 9.589e-17, 2.877e-17]), # Belly and Van Regermorter neutral
        CE(j=13, i=2, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[9.589e-16, 5.753e-16, 4.110e-16, 2.877e-16, 1.438e-16, 9.589e-17, 2.877e-17]), # Belly and Van Regermorter neutral
		CE(j=14, i=2, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[9.589e-16, 5.753e-16, 4.110e-16, 2.877e-16, 1.438e-16, 9.589e-17, 2.877e-17]), # Belly and Van Regermorter neutral
        CE(j=4, i=3, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[5.753e-16, 3.452e-16, 2.466e-16, 1.726e-16, 8.630e-17, 5.753e-17, 1.726e-17]), # Belly and Van Regermorter neutral
		CE(j=5, i=3, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[5.753e-16, 3.452e-16, 2.466e-16, 1.726e-16, 8.630e-17, 5.753e-17, 1.726e-17]), # Belly and Van Regermorter neutral
		CE(j=6, i=3, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[5.753e-16, 3.452e-16, 2.466e-16, 1.726e-16, 8.630e-17, 5.753e-17, 1.726e-17]), # Belly and Van Regermorter neutral
		CE(j=7, i=3, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[5.753e-16, 3.452e-16, 2.466e-16, 1.726e-16, 8.630e-17, 5.753e-17, 1.726e-17]), # Belly and Van Regermorter neutral
		CE(j=8, i=3, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[5.753e-16, 3.452e-16, 2.466e-16, 1.726e-16, 8.630e-17, 5.753e-17, 1.726e-17]), # Belly and Van Regermorter neutral
		CE(j=9, i=3, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[5.753e-16, 3.452e-16, 2.466e-16, 1.726e-16, 8.630e-17, 5.753e-17, 1.726e-17]), # Belly and Van Regermorter neutral
		CE(j=10, i=3, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[5.753e-16, 3.452e-16, 2.466e-16, 1.726e-16, 8.630e-17, 5.753e-17, 1.726e-17]), # Belly and Van Regermorter neutral
		CE(j=11, i=3, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[5.753e-16, 3.452e-16, 2.466e-16, 1.726e-16, 8.630e-17, 5.753e-17, 1.726e-17]), # Belly and Van Regermorter neutral
		CE(j=12, i=3, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[5.753e-16, 3.452e-16, 2.466e-16, 1.726e-16, 8.630e-17, 5.753e-17, 1.726e-17]), # Belly and Van Regermorter neutral
		CE(j=13, i=3, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[5.753e-16, 3.452e-16, 2.466e-16, 1.726e-16, 8.630e-17, 5.753e-17, 1.726e-17]), # Belly and Van Regermorter neutral
		CE(j=14, i=3, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[5.753e-16, 3.452e-16, 2.466e-16, 1.726e-16, 8.630e-17, 5.753e-17, 1.726e-17]), # Belly and Van Regermorter neutral
		CE(j=5, i=4, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[9.589e-16, 5.753e-16, 4.110e-16, 2.877e-16, 1.438e-16, 9.589e-17, 2.877e-17]), # Belly and Van Regermorter neutral
		CE(j=6, i=4, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[9.589e-16, 5.753e-16, 4.110e-16, 2.877e-16, 1.438e-16, 9.589e-17, 2.877e-17]), # Belly and Van Regermorter neutral
		CE(j=7, i=4, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[9.589e-16, 5.753e-16, 4.110e-16, 2.877e-16, 1.438e-16, 9.589e-17, 2.877e-17]), # Belly and Van Regermorter neutral
		CE(j=8, i=4, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[9.589e-16, 5.753e-16, 4.110e-16, 2.877e-16, 1.438e-16, 9.589e-17, 2.877e-17]), # Belly and Van Regermorter neutral
		CE(j=9, i=4, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[9.589e-16, 5.753e-16, 4.110e-16, 2.877e-16, 1.438e-16, 9.589e-17, 2.877e-17]), # Belly and Van Regermorter neutral
		CE(j=10, i=4, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[9.589e-16, 5.753e-16, 4.110e-16, 2.877e-16, 1.438e-16, 9.589e-17, 2.877e-17]), # Belly and Van Regermorter neutral
		CE(j=11, i=4, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[9.589e-16, 5.753e-16, 4.110e-16, 2.877e-16, 1.438e-16, 9.589e-17, 2.877e-17]), # Belly and Van Regermorter neutral
		CE(j=12, i=4, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[9.589e-16, 5.753e-16, 4.110e-16, 2.877e-16, 1.438e-16, 9.589e-17, 2.877e-17]), # Belly and Van Regermorter neutral
		CE(j=13, i=4, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[9.589e-16, 5.753e-16, 4.110e-16, 2.877e-16, 1.438e-16, 9.589e-17, 2.877e-17]), # Belly and Van Regermorter neutral
		CE(j=14, i=4, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[9.589e-16, 5.753e-16, 4.110e-16, 2.877e-16, 1.438e-16, 9.589e-17, 2.877e-17]), # Belly and Van Regermorter neutral
		CE(j=6, i=5, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[5.753e-16, 3.452e-16, 2.466e-16, 1.726e-16, 8.630e-17, 5.753e-17, 1.726e-17]), # Belly and Van Regermorter neutral
		CE(j=7, i=5, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[5.753e-16, 3.452e-16, 2.466e-16, 1.726e-16, 8.630e-17, 5.753e-17, 1.726e-17]), # Belly and Van Regermorter neutral
		CE(j=8, i=5, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[5.753e-16, 3.452e-16, 2.466e-16, 1.726e-16, 8.630e-17, 5.753e-17, 1.726e-17]), # Belly and Van Regermorter neutral
		CE(j=9, i=5, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[5.753e-16, 3.452e-16, 2.466e-16, 1.726e-16, 8.630e-17, 5.753e-17, 1.726e-17]), # Belly and Van Regermorter neutral
		CE(j=10, i=5, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[5.753e-16, 3.452e-16, 2.466e-16, 1.726e-16, 8.630e-17, 5.753e-17, 1.726e-17]), # Belly and Van Regermorter neutral
		CE(j=11, i=5, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[5.753e-16, 3.452e-16, 2.466e-16, 1.726e-16, 8.630e-17, 5.753e-17, 1.726e-17]), # Belly and Van Regermorter neutral
		CE(j=12, i=5, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[5.753e-16, 3.452e-16, 2.466e-16, 1.726e-16, 8.630e-17, 5.753e-17, 1.726e-17]), # Belly and Van Regermorter neutral
		CE(j=13, i=5, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[5.753e-16, 3.452e-16, 2.466e-16, 1.726e-16, 8.630e-17, 5.753e-17, 1.726e-17]), # Belly and Van Regermorter neutral
		CE(j=14, i=5, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[5.753e-16, 3.452e-16, 2.466e-16, 1.726e-16, 8.630e-17, 5.753e-17, 1.726e-17]), # Belly and Van Regermorter neutral
		CE(j=7, i=6, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[4.110e-16, 2.466e-16, 1.761e-16, 1.233e-16, 6.164e-17, 4.110e-17, 1.233e-17]), # Belly and Van Regermorter neutral
		CE(j=8, i=6, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[4.110e-16, 2.466e-16, 1.761e-16, 1.233e-16, 6.164e-17, 4.110e-17, 1.233e-17]), # Belly and Van Regermorter neutral
		CE(j=9, i=6, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[4.110e-16, 2.466e-16, 1.761e-16, 1.233e-16, 6.164e-17, 4.110e-17, 1.233e-17]), # Belly and Van Regermorter neutral
		CE(j=10, i=6, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[4.110e-16, 2.466e-16, 1.761e-16, 1.233e-16, 6.164e-17, 4.110e-17, 1.233e-17]), # Belly and Van Regermorter neutral
		CE(j=11, i=6, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[4.110e-16, 2.466e-16, 1.761e-16, 1.233e-16, 6.164e-17, 4.110e-17, 1.233e-17]), # Belly and Van Regermorter neutral
		CE(j=12, i=6, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[4.110e-16, 2.466e-16, 1.761e-16, 1.233e-16, 6.164e-17, 4.110e-17, 1.233e-17]), # Belly and Van Regermorter neutral
		CE(j=13, i=6, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[4.110e-16, 2.466e-16, 1.761e-16, 1.233e-16, 6.164e-17, 4.110e-17, 1.233e-17]), # Belly and Van Regermorter neutral
		CE(j=14, i=6, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[4.110e-16, 2.466e-16, 1.761e-16, 1.233e-16, 6.164e-17, 4.110e-17, 1.233e-17]), # Belly and Van Regermorter neutral
		CE(j=8, i=7, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[5.753e-16, 3.452e-16, 2.466e-16, 1.726e-16, 8.630e-17, 5.753e-17, 1.726e-17]), # Belly and Van Regermorter neutral
		CE(j=9, i=7, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[5.753e-16, 3.452e-16, 2.466e-16, 1.726e-16, 8.630e-17, 5.753e-17, 1.726e-17]), # Belly and Van Regermorter neutral
		CE(j=10, i=7, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[4.571e-16, 3.385e-16, 3.208e-16, 2.665e-16, 2.115e-16, 2.261e-16, 1.400e-16]), # Van Regermorter neutral
		CE(j=11, i=7, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[5.753e-16, 3.452e-16, 2.466e-16, 1.726e-16, 8.630e-17, 5.753e-17, 1.726e-17]), # Belly and Van Regermorter neutral
		CE(j=12, i=7, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[5.753e-16, 3.452e-16, 2.466e-16, 1.726e-16, 8.630e-17, 5.753e-17, 1.726e-17]), # Belly and Van Regermorter neutral
		CE(j=13, i=7, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[8.852e-16, 7.552e-16, 6.099e-16, 5.731e-16, 4.538e-16, 3.994e-16, 3.006e-16]), # Van Regermorter neutral
		CE(j=14, i=7, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[5.753e-16, 3.452e-16, 2.466e-16, 1.726e-16, 8.630e-17, 5.753e-17, 1.726e-17]), # Belly and Van Regermorter neutral
		CE(j=9, i=8, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[1.123e-16, 9.711e-17, 8.505e-17, 7.695e-17, 7.100e-17, 5.842e-17, 3.843e-17]), # Van Regermorter neutral
		CE(j=10, i=8, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[9.589e-16, 5.753e-16, 4.110e-16, 2.877e-16, 1.438e-16, 9.589e-17, 2.877e-17]), # Belly and Van Regermorter neutral
		CE(j=11, i=8, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[1.147e-15, 8.773e-16, 8.066e-16, 6.918e-16, 5.697e-16, 5.733e-16, 3.593e-16]), # Van Regermorter neutral
		CE(j=12, i=8, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[9.589e-16, 5.753e-16, 4.110e-16, 2.877e-16, 1.438e-16, 9.589e-17, 2.877e-17]), # Belly and Van Regermorter neutral
		CE(j=13, i=8, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[9.589e-16, 5.753e-16, 4.110e-16, 2.877e-16, 1.438e-16, 9.589e-17, 2.877e-17]), # Belly and Van Regermorter neutral
		CE(j=14, i=8, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[2.910e-16, 2.384e-16, 2.016e-16, 1.828e-16, 1.448e-16, 1.351e-16, 9.591e-17]), # Van Regermorter neutral
		CE(j=10, i=9, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[2.877e-15, 1.726e-15, 1.233e-15, 8.630e-16, 4.315e-16, 2.877e-16, 8.630e-17]), # Belly and Van Regermorter neutral
		CE(j=11, i=9, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[2.877e-15, 1.726e-15, 1.233e-15, 8.630e-16, 4.315e-16, 2.877e-16, 8.630e-17]), # Belly and Van Regermorter neutral
		CE(j=12, i=9, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[5.838e-16, 5.104e-16, 4.598e-16, 4.047e-16, 3.774e-16, 3.051e-16, 2.012e-16]), # Van Regermorter neutral
		CE(j=13, i=9, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[2.877e-15, 1.726e-15, 1.233e-15, 8.630e-16, 4.315e-16, 2.877e-16, 8.630e-17]), # Belly and Van Regermorter neutral
		CE(j=14, i=9, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[2.877e-15, 1.726e-15, 1.233e-15, 8.630e-16, 4.315e-16, 2.877e-16, 8.630e-17]), # Belly and Van Regermorter neutral
		CE(j=11, i=10, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[9.589e-16, 5.753e-16, 4.110e-16, 2.877e-16, 1.438e-16, 9.589e-17, 2.877e-17]), # Belly and Van Regermorter neutral
		CE(j=12, i=10, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[9.589e-16, 5.753e-16, 4.110e-16, 2.877e-16, 1.438e-16, 9.589e-17, 2.877e-17]), # Belly and Van Regermorter neutral
		CE(j=13, i=10, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[9.589e-16, 5.753e-16, 4.110e-16, 2.877e-16, 1.438e-16, 9.589e-17, 2.877e-17]), # Belly and Van Regermorter neutral
		CE(j=14, i=10, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[4.138e-15, 3.481e-15, 3.040e-15, 3.037e-15, 2.351e-15, 2.171e-15, 1.303e-15]), # Van Regermorter neutral
		CE(j=12, i=11, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[5.753e-16, 3.452e-16, 2.466e-16, 1.726e-16, 8.630e-17, 5.753e-17, 1.726e-17]), # Belly and Van Regermorter neutral
		CE(j=13, i=11, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[1.981e-14, 1.714e-14, 1.456e-14, 1.558e-14, 1.209e-14, 1.040e-14, 6.285e-15]), # Van Regermorter neutral
		CE(j=14, i=11, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[5.753e-16, 3.452e-16, 2.466e-16, 1.726e-16, 8.630e-17, 5.753e-17, 1.726e-17]), # Belly and Van Regermorter neutral
		CE(j=13, i=12, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[9.589e-16, 5.753e-16, 4.110e-16, 2.877e-16, 1.438e-16, 9.589e-17, 2.877e-17]), # Belly and Van Regermorter neutral
		CE(j=14, i=12, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[9.589e-16, 5.753e-16, 4.110e-16, 2.877e-16, 1.438e-16, 9.589e-17, 2.877e-17]), # Belly and Van Regermorter neutral
		CE(j=14, i=13, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[4.110e-16, 2.466e-16, 1.761e-16, 1.233e-16, 6.164e-17, 4.110e-17, 1.233e-17]), # Belly and Van Regermorter neutral
		CI(j=0, i=15, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[2.880e-16, 6.468e-16, 8.122e-16, 9.197e-16, 1.020e-15, 1.034e-15, 8.879e-16]),
		CI(j=1, i=15, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[1.179e-15, 1.744e-15, 1.949e-15, 2.084e-15, 2.183e-15, 2.162e-15, 1.556e-15]),
		CI(j=2, i=15, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[1.202e-15, 1.769e-15, 1.975e-15, 2.111e-15, 2.209e-15, 2.187e-15, 1.568e-15]),
		CI(j=3, i=15, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[1.251e-15, 1.824e-15, 2.032e-15, 2.168e-15, 2.266e-15, 2.241e-15, 1.596e-15]),
		CI(j=4, i=15, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[1.756e-15, 2.378e-15, 2.599e-15, 2.744e-15, 2.827e-15, 2.778e-15, 1.848e-15]),
		CI(j=5, i=15, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[1.768e-15, 2.390e-15, 2.612e-15, 2.756e-15, 2.839e-15, 2.790e-15, 1.853e-15]),
		CI(j=6, i=15, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[1.786e-15, 2.410e-15, 2.633e-15, 2.778e-15, 2.860e-15, 2.810e-15, 1.862e-15]),
		CI(j=7, i=15, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[2.175e-15, 2.827e-15, 3.059e-15, 3.208e-15, 3.276e-15, 3.205e-15, 2.030e-15]),
		CI(j=8, i=15, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[2.582e-15, 3.258e-15, 3.499e-15, 3.651e-15, 3.701e-15, 3.607e-15, 2.189e-15]),
		CI(j=9, i=15, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[8.005e-15, 8.887e-15, 9.175e-15, 9.274e-15, 8.951e-15, 8.255e-15, 3.533e-15]),
		CI(j=10, i=15, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[1.433e-14, 1.534e-14, 1.557e-14, 1.551e-14, 1.441e-14, 1.254e-14, 4.380e-15]),
		CI(j=11, i=15, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[1.616e-14, 1.717e-14, 1.738e-14, 1.726e-14, 1.587e-14, 1.362e-14, 4.560e-15]),
		CI(j=12, i=15, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[4.355e-14, 4.419e-14, 4.365e-14, 4.220e-14, 3.352e-14, 2.532e-14, 6.049e-15]),
		CI(j=13, i=15, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[5.291e-14, 5.326e-14, 5.238e-14, 5.017e-14, 3.830e-14, 2.820e-14, 6.328e-15]),
		CI(j=14, i=15, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[5.637e-14, 5.660e-14, 5.558e-14, 5.305e-14, 3.996e-14, 2.918e-14, 6.417e-15]),

    ]
    )




def sr_rh_simple():
    """
    Custom Sr I atomic model based on data/StrontiumI_LR.atom BUT with the values from
    lightweaver/RH.

    Contains 13 levels (12 Sr I + 1 Sr II ground state) and the subset of RH
    VoigtLines whose both upper and lower levels are present in this model.
    Level indices remapped from RH: RH-17 -> local-10, RH-18 -> local-11.

    Returns
    -------
    AtomicModel
        Sr I atomic model
    """
    return AtomicModel(element = Element(Z=38),
    levels = [
        AtomicLevel(E=     0.000, g=1, label="SR I 4P6 5S2 1SE 0",  stage=0, J=Fraction(0, 1), L=0, S=Fraction(0, 1)),
        AtomicLevel(E= 14317.520, g=1, label="SR I 4P6 5S5P 3PO 0", stage=0, J=Fraction(0, 1), L=1, S=Fraction(1, 1)),
        AtomicLevel(E= 14504.351, g=3, label="SR I 4P6 5S5P 3PO 1", stage=0, J=Fraction(1, 1), L=1, S=Fraction(1, 1)),
        AtomicLevel(E= 14898.563, g=5, label="SR I 4P6 5S5P 3PO 2", stage=0, J=Fraction(2, 1), L=1, S=Fraction(1, 1)),
        AtomicLevel(E= 18159.057, g=3, label="SR I 4P6 5S4D 3DE 1", stage=0, J=Fraction(1, 1), L=2, S=Fraction(1, 1)),
        AtomicLevel(E= 18218.795, g=5, label="SR I 4P6 5S4D 3DE 2", stage=0, J=Fraction(2, 1), L=2, S=Fraction(1, 1)),
        AtomicLevel(E= 18319.268, g=7, label="SR I 4P6 5S4D 3DE 3", stage=0, J=Fraction(3, 1), L=2, S=Fraction(1, 1)),
        AtomicLevel(E= 20149.699, g=5, label="SR I 4P6 5S4D 1DE 2", stage=0, J=Fraction(2, 1), L=2, S=Fraction(0, 1)),
        AtomicLevel(E= 21698.482, g=3, label="SR I 4P6 5S5P 1PO 1", stage=0, J=Fraction(1, 1), L=1, S=Fraction(0, 1)),
        AtomicLevel(E= 29038.795, g=3, label="SR I 4P6 5S6S 3SE 1", stage=0, J=Fraction(1, 1), L=0, S=Fraction(1, 1)),
        AtomicLevel(E= 34098.441, g=3, label="SR I 4P6 5S6P 1PO 1", stage=0, J=Fraction(1, 1), L=1, S=Fraction(0, 1)),
        AtomicLevel(E= 34727.484, g=5, label="SR I 4P6 5S5D 1DE 2", stage=0, J=Fraction(2, 1), L=2, S=Fraction(0, 1)),
        AtomicLevel(E= 45932.090, g=2, label="SR II 4P6 5S2 2SE 1", stage=1, J=Fraction(1, 2), L=0, S=Fraction(1, 2)),
    ],
    lines = [
        VoigtLine(j=10, i=0, f=7.079e-03, type=LineType.CRD, quadrature=LinearCoreExpWings(qCore=1, qWing=20, Nlambda=10), broadening=LineBroadening(natural=[RadiativeBroadening(gamma=7.41e+06)], elastic=[VdwUnsold(vals=[1.0, 1.0]), MultiplicativeStarkBroadening(coeff=0)])),
        VoigtLine(j=8,  i=0, f=2.692e-01, type=LineType.CRD, quadrature=LinearCoreExpWings(qCore=3, qWing=100, Nlambda=50), broadening=LineBroadening(natural=[RadiativeBroadening(gamma=2.82e+07)], elastic=[VdwUnsold(vals=[1.0, 1.0]), MultiplicativeStarkBroadening(coeff=0)])),
        VoigtLine(j=1,  i=0, f=2.692e-01, type=LineType.CRD, quadrature=LinearCoreExpWings(qCore=3, qWing=100, Nlambda=50), broadening=LineBroadening(natural=[RadiativeBroadening(gamma=2.82e+07)], elastic=[VdwUnsold(vals=[1.0, 1.0]), MultiplicativeStarkBroadening(coeff=0)])),
        VoigtLine(j=9,  i=1, f=1.905e-01, type=LineType.CRD, quadrature=LinearCoreExpWings(qCore=1, qWing=20, Nlambda=10), broadening=LineBroadening(natural=[RadiativeBroadening(gamma=6.32e+07)], elastic=[VdwUnsold(vals=[1.0, 1.0]), MultiplicativeStarkBroadening(coeff=0)])),
        VoigtLine(j=9,  i=2, f=1.749e-01, type=LineType.CRD, quadrature=LinearCoreExpWings(qCore=1, qWing=20, Nlambda=10), broadening=LineBroadening(natural=[RadiativeBroadening(gamma=6.32e+07)], elastic=[VdwUnsold(vals=[1.0, 1.0]), MultiplicativeStarkBroadening(coeff=0)])),
        VoigtLine(j=2,  i=0, f=1.445e-03, type=LineType.CRD, quadrature=LinearCoreExpWings(qCore=1, qWing=20, Nlambda=10), broadening=LineBroadening(natural=[RadiativeBroadening(gamma=67600)],    elastic=[VdwUnsold(vals=[1.0, 1.0]), MultiplicativeStarkBroadening(coeff=0)])),
        VoigtLine(j=9,  i=3, f=1.321e-01, type=LineType.CRD, quadrature=LinearCoreExpWings(qCore=1, qWing=20, Nlambda=10), broadening=LineBroadening(natural=[RadiativeBroadening(gamma=6.32e+07)], elastic=[VdwUnsold(vals=[1.0, 1.0]), MultiplicativeStarkBroadening(coeff=0)])),
        VoigtLine(j=10, i=7, f=2.576e-02, type=LineType.CRD, quadrature=LinearCoreExpWings(qCore=1, qWing=20, Nlambda=10), broadening=LineBroadening(natural=[RadiativeBroadening(gamma=7.41e+06)], elastic=[VdwUnsold(vals=[1.0, 1.0]), MultiplicativeStarkBroadening(coeff=0)])),
        VoigtLine(j=11, i=8, f=1.007e-01, type=LineType.CRD, quadrature=LinearCoreExpWings(qCore=1, qWing=20, Nlambda=10), broadening=LineBroadening(natural=[RadiativeBroadening(gamma=6.84e+06)], elastic=[VdwUnsold(vals=[1.0, 1.0]), MultiplicativeStarkBroadening(coeff=0)])),
    ],
    continua = [
      HydrogenicContinuum(j=12, i=0, NlambdaGen=15, alpha0=6.025e-22, minWavelength=52.3),
    ],
    collisions = [	
      CE(j=1, i=0, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[9.386e-16, 7.97e-16, 7.157e-16, 6.385e-16, 5.115e-16, 4.492e-16, 3.056e-16]),
		CE(j=2, i=0, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[2.285e-16, 1.394e-16, 1.012e-16, 7.246e-17, 3.869e-17, 2.721e-17, 1.016e-17]),
		CE(j=3, i=0, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[8.779e-16, 7.455e-16, 6.694e-16, 5.972e-16, 4.784e-16, 4.202e-16, 2.858e-16]),
		CE(j=4, i=0, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[6.296e-16, 5.346e-16, 4.801e-16, 4.283e-16, 3.431e-16, 3.013e-16, 2.05e-16]),
		CE(j=5, i=0, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[6.261e-16, 5.317e-16, 4.774e-16, 4.259e-16, 3.412e-16, 2.997e-16, 2.039e-16]),
		CE(j=6, i=0, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[6.204e-16, 5.268e-16, 4.73e-16, 4.22e-16, 3.381e-16, 2.969e-16, 2.02e-16]),
		CE(j=7, i=0, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[5.286e-16, 4.489e-16, 4.031e-16, 3.596e-16, 2.881e-16, 2.53e-16, 1.721e-16]),
		CE(j=8, i=0, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[2.145e-14, 1.319e-14, 9.648e-15, 6.91e-15, 3.689e-15, 2.602e-15, 9.799e-16]),
		CE(j=9, i=0, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[2.861e-16, 2.43e-16, 2.182e-16, 1.946e-16, 1.559e-16, 1.369e-16, 9.316e-17]),
		CE(j=10, i=0, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[2.277e-16, 1.934e-16, 1.736e-16, 1.549e-16, 1.241e-16, 1.09e-16, 7.414e-17]),
		CE(j=11, i=0, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[2.24e-16, 1.902e-16, 1.708e-16, 1.524e-16, 1.221e-16, 1.072e-16, 7.294e-17]),
		CE(j=12, i=0, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[2.214e-16, 1.88e-16, 1.688e-16, 1.506e-16, 1.206e-16, 1.06e-16, 7.209e-17]),
# 		CE(j=13, i=0, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[2.211e-16, 1.878e-16, 1.686e-16, 1.504e-16, 1.205e-16, 1.058e-16, 7.199e-17]),
# 		CE(j=14, i=0, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[2.209e-16, 1.876e-16, 1.685e-16, 1.503e-16, 1.204e-16, 1.058e-16, 7.194e-17]),
# 		CE(j=15, i=0, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[2.204e-16, 1.872e-16, 1.68e-16, 1.499e-16, 1.201e-16, 1.055e-16, 7.176e-17]),
# 		CE(j=16, i=0, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[2.198e-16, 1.867e-16, 1.676e-16, 1.495e-16, 1.198e-16, 1.052e-16, 7.157e-17]),
# 		CE(j=17, i=0, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[2.745e-16, 1.664e-16, 1.201e-16, 8.536e-17, 4.479e-17, 3.12e-17, 1.165e-17]),
# 		CE(j=18, i=0, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[2.118e-16, 1.799e-16, 1.615e-16, 1.441e-16, 1.154e-16, 1.014e-16, 6.897e-17]),
# 		CE(j=19, i=0, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[2.09e-16, 1.775e-16, 1.594e-16, 1.422e-16, 1.139e-16, 1e-16, 6.805e-17]),
# 		CE(j=20, i=0, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[2.089e-16, 1.774e-16, 1.593e-16, 1.421e-16, 1.138e-16, 9.996e-17, 6.8e-17]),
# 		CE(j=21, i=0, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[2.086e-16, 1.772e-16, 1.591e-16, 1.419e-16, 1.137e-16, 9.985e-17, 6.793e-17]),
# 		CE(j=22, i=0, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[2.071e-16, 1.759e-16, 1.58e-16, 1.409e-16, 1.129e-16, 9.915e-17, 6.745e-17]),
# 		CE(j=23, i=0, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[2.051e-16, 1.742e-16, 1.564e-16, 1.395e-16, 1.118e-16, 9.818e-17, 6.679e-17]),
# 		CE(j=24, i=0, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[2.025e-16, 1.719e-16, 1.544e-16, 1.377e-16, 1.103e-16, 9.691e-17, 6.592e-17]),
# 		CE(j=25, i=0, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[1.97e-16, 1.673e-16, 1.502e-16, 1.34e-16, 1.073e-16, 9.428e-17, 6.413e-17]),
# 		CE(j=26, i=0, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[1.959e-16, 1.664e-16, 1.494e-16, 1.333e-16, 1.068e-16, 9.377e-17, 6.379e-17]),
# 		CE(j=27, i=0, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[1.943e-16, 1.65e-16, 1.482e-16, 1.322e-16, 1.059e-16, 9.3e-17, 6.327e-17]),
# 		CE(j=28, i=0, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[1.908e-16, 1.62e-16, 1.455e-16, 1.298e-16, 1.04e-16, 9.131e-17, 6.212e-17]),
# 		CE(j=29, i=0, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[1.891e-16, 1.606e-16, 1.442e-16, 1.286e-16, 1.03e-16, 9.049e-17, 6.156e-17]),
# 		CE(j=30, i=0, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[1.879e-16, 1.596e-16, 1.433e-16, 1.278e-16, 1.024e-16, 8.995e-17, 6.119e-17]),
# 		CE(j=31, i=0, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[1.878e-16, 1.595e-16, 1.432e-16, 1.278e-16, 1.024e-16, 8.991e-17, 6.116e-17]),
# 		CE(j=32, i=0, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[1.876e-16, 1.593e-16, 1.43e-16, 1.276e-16, 1.022e-16, 8.977e-17, 6.107e-17]),
 		CE(j=2, i=1, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[1.375e-12, 1.168e-12, 1.048e-12, 9.354e-13, 7.493e-13, 6.581e-13, 4.477e-13]),
		CE(j=3, i=1, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[2.044e-13, 1.736e-13, 1.559e-13, 1.39e-13, 1.114e-13, 9.783e-14, 6.655e-14]),
		CE(j=4, i=1, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[8.558e-15, 7.267e-15, 6.526e-15, 5.822e-15, 4.664e-15, 4.096e-15, 2.786e-15]),
		CE(j=5, i=1, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[8.339e-15, 7.081e-15, 6.359e-15, 5.673e-15, 4.544e-15, 3.991e-15, 2.715e-15]),
		CE(j=6, i=1, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[7.99e-15, 6.785e-15, 6.093e-15, 5.435e-15, 4.354e-15, 3.824e-15, 2.602e-15]),
		CE(j=7, i=1, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[4.244e-15, 3.604e-15, 3.236e-15, 2.887e-15, 2.313e-15, 2.031e-15, 1.382e-15]),
		CE(j=8, i=1, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[2.857e-15, 2.426e-15, 2.178e-15, 1.943e-15, 1.557e-15, 1.367e-15, 9.302e-16]),
		CE(j=9, i=1, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[1.975e-14, 1.211e-14, 8.835e-15, 6.372e-15, 3.472e-15, 2.482e-15, 9.878e-16]),
		CE(j=10, i=1, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[5.861e-16, 4.977e-16, 4.469e-16, 3.987e-16, 3.194e-16, 2.805e-16, 1.908e-16]),
		CE(j=11, i=1, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[5.697e-16, 4.838e-16, 4.344e-16, 3.876e-16, 3.105e-16, 2.727e-16, 1.855e-16]),
		CE(j=12, i=1, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[5.581e-16, 4.74e-16, 4.256e-16, 3.797e-16, 3.041e-16, 2.671e-16, 1.817e-16]),
# 		CE(j=13, i=1, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[5.568e-16, 4.729e-16, 4.246e-16, 3.788e-16, 3.034e-16, 2.665e-16, 1.813e-16]),
# 		CE(j=14, i=1, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[5.561e-16, 4.723e-16, 4.241e-16, 3.783e-16, 3.031e-16, 2.662e-16, 1.811e-16]),
# 		CE(j=15, i=1, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[5.537e-16, 4.702e-16, 4.222e-16, 3.767e-16, 3.017e-16, 2.65e-16, 1.803e-16]),
# 		CE(j=16, i=1, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[5.512e-16, 4.68e-16, 4.203e-16, 3.749e-16, 3.004e-16, 2.638e-16, 1.795e-16]),
# 		CE(j=17, i=1, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[5.453e-16, 4.631e-16, 4.158e-16, 3.71e-16, 2.972e-16, 2.61e-16, 1.775e-16]),
# 		CE(j=18, i=1, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[5.174e-16, 4.393e-16, 3.945e-16, 3.519e-16, 2.819e-16, 2.476e-16, 1.685e-16]),
# 		CE(j=19, i=1, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[5.057e-16, 4.294e-16, 3.856e-16, 3.44e-16, 2.756e-16, 2.42e-16, 1.646e-16]),
# 		CE(j=20, i=1, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[5.051e-16, 4.289e-16, 3.851e-16, 3.436e-16, 2.752e-16, 2.417e-16, 1.644e-16]),
# 		CE(j=21, i=1, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[5.041e-16, 4.281e-16, 3.844e-16, 3.429e-16, 2.747e-16, 2.413e-16, 1.641e-16]),
# 		CE(j=22, i=1, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[4.981e-16, 4.23e-16, 3.798e-16, 3.389e-16, 2.714e-16, 2.384e-16, 1.622e-16]),
# 		CE(j=23, i=1, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[2.627e-14, 1.605e-14, 1.167e-14, 8.382e-15, 4.531e-15, 3.229e-15, 1.299e-15]),
# 		CE(j=24, i=1, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[4.794e-16, 4.071e-16, 3.656e-16, 3.261e-16, 2.612e-16, 2.295e-16, 1.561e-16]),
# 		CE(j=25, i=1, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[4.58e-16, 3.889e-16, 3.492e-16, 3.115e-16, 2.496e-16, 2.192e-16, 1.491e-16]),
# 		CE(j=26, i=1, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[4.539e-16, 3.854e-16, 3.461e-16, 3.088e-16, 2.473e-16, 2.172e-16, 1.478e-16]),
# 		CE(j=27, i=1, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[4.478e-16, 3.803e-16, 3.415e-16, 3.046e-16, 2.44e-16, 2.143e-16, 1.458e-16]),
# 		CE(j=28, i=1, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[4.345e-16, 3.69e-16, 3.313e-16, 2.956e-16, 2.368e-16, 2.08e-16, 1.415e-16]),
# 		CE(j=29, i=1, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[4.282e-16, 3.636e-16, 3.265e-16, 2.913e-16, 2.333e-16, 2.049e-16, 1.394e-16]),
# 		CE(j=30, i=1, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[4.241e-16, 3.601e-16, 3.234e-16, 2.885e-16, 2.311e-16, 2.03e-16, 1.381e-16]),
# 		CE(j=31, i=1, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[4.237e-16, 3.598e-16, 3.231e-16, 2.883e-16, 2.309e-16, 2.028e-16, 1.38e-16]),
# 		CE(j=32, i=1, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[4.227e-16, 3.59e-16, 3.223e-16, 2.875e-16, 2.303e-16, 2.023e-16, 1.376e-16]),
		CE(j=3, i=2, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[3.922e-13, 3.331e-13, 2.991e-13, 2.668e-13, 2.137e-13, 1.877e-13, 1.277e-13]),
		CE(j=4, i=2, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[9.306e-15, 7.902e-15, 7.096e-15, 6.33e-15, 5.071e-15, 4.454e-15, 3.03e-15]),
		CE(j=5, i=2, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[9.056e-15, 7.69e-15, 6.905e-15, 6.16e-15, 4.935e-15, 4.334e-15, 2.948e-15]),
		CE(j=6, i=2, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[8.658e-15, 7.353e-15, 6.602e-15, 5.89e-15, 4.718e-15, 4.144e-15, 2.819e-15]),
		CE(j=7, i=2, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[4.482e-15, 3.806e-15, 3.418e-15, 3.049e-15, 2.443e-15, 2.145e-15, 1.459e-15]),
		CE(j=8, i=2, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[2.983e-15, 2.533e-15, 2.274e-15, 2.029e-15, 1.625e-15, 1.428e-15, 9.712e-16]),
		CE(j=9, i=2, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[1.84e-14, 1.129e-14, 8.235e-15, 5.941e-15, 3.239e-15, 2.316e-15, 9.226e-16]),
		CE(j=10, i=2, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[5.959e-16, 5.061e-16, 4.544e-16, 4.054e-16, 3.248e-16, 2.852e-16, 1.94e-16]),
		CE(j=11, i=2, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[5.791e-16, 4.918e-16, 4.416e-16, 3.939e-16, 3.156e-16, 2.772e-16, 1.886e-16]),
		CE(j=12, i=2, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[5.672e-16, 4.817e-16, 4.325e-16, 3.859e-16, 3.091e-16, 2.715e-16, 1.847e-16]),
# 		CE(j=13, i=2, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[5.659e-16, 4.806e-16, 4.315e-16, 3.85e-16, 3.084e-16, 2.709e-16, 1.843e-16]),
# 		CE(j=14, i=2, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[5.652e-16, 4.799e-16, 4.31e-16, 3.845e-16, 3.08e-16, 2.705e-16, 1.84e-16]),
# 		CE(j=15, i=2, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[5.627e-16, 4.778e-16, 4.291e-16, 3.828e-16, 3.066e-16, 2.693e-16, 1.832e-16]),
# 		CE(j=16, i=2, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[5.601e-16, 4.756e-16, 4.271e-16, 3.81e-16, 3.052e-16, 2.681e-16, 1.824e-16]),
# 		CE(j=17, i=2, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[5.541e-16, 4.705e-16, 4.225e-16, 3.769e-16, 3.019e-16, 2.652e-16, 1.804e-16]),
# 		CE(j=18, i=2, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[5.254e-16, 4.462e-16, 4.006e-16, 3.574e-16, 2.863e-16, 2.515e-16, 1.711e-16]),
# 		CE(j=19, i=2, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[2.403e-15, 1.469e-15, 1.068e-15, 7.676e-16, 4.153e-16, 2.961e-16, 1.192e-16]),
# 		CE(j=20, i=2, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[1.203e-14, 7.351e-15, 5.346e-15, 3.842e-15, 2.079e-15, 1.482e-15, 5.966e-16]),
# 		CE(j=21, i=2, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[5.118e-16, 4.347e-16, 3.903e-16, 3.482e-16, 2.789e-16, 2.45e-16, 1.667e-16]),
# 		CE(j=22, i=2, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[1.457e-14, 8.905e-15, 6.477e-15, 4.654e-15, 2.517e-15, 1.795e-15, 7.226e-16]),
# 		CE(j=23, i=2, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[5.697e-15, 3.481e-15, 2.532e-15, 1.819e-15, 9.837e-16, 7.013e-16, 2.824e-16]),
# 		CE(j=24, i=2, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[1.335e-14, 8.154e-15, 5.929e-15, 4.259e-15, 2.303e-15, 1.641e-15, 6.611e-16]),
# 		CE(j=25, i=2, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[4.646e-16, 3.945e-16, 3.543e-16, 3.16e-16, 2.532e-16, 2.224e-16, 1.513e-16]),
# 		CE(j=26, i=2, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[4.604e-16, 3.91e-16, 3.511e-16, 3.132e-16, 2.509e-16, 2.204e-16, 1.499e-16]),
# 		CE(j=27, i=2, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[4.542e-16, 3.857e-16, 3.463e-16, 3.09e-16, 2.475e-16, 2.174e-16, 1.479e-16]),
# 		CE(j=28, i=2, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[4.406e-16, 3.742e-16, 3.36e-16, 2.997e-16, 2.401e-16, 2.109e-16, 1.435e-16]),
# 		CE(j=29, i=2, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[4.341e-16, 3.687e-16, 3.31e-16, 2.953e-16, 2.366e-16, 2.078e-16, 1.414e-16]),
# 		CE(j=30, i=2, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[4.299e-16, 3.651e-16, 3.278e-16, 2.925e-16, 2.343e-16, 2.058e-16, 1.4e-16]),
# 		CE(j=31, i=2, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[4.296e-16, 3.648e-16, 3.276e-16, 2.922e-16, 2.341e-16, 2.056e-16, 1.399e-16]),
#		CE(j=32, i=2, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[4.285e-16, 3.639e-16, 3.268e-16, 2.915e-16, 2.335e-16, 2.051e-16, 1.395e-16]),
		CE(j=4, i=3, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[1.127e-14, 9.573e-15, 8.595e-15, 7.668e-15, 6.143e-15, 5.395e-15, 3.67e-15]),
		CE(j=5, i=3, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[1.093e-14, 9.285e-15, 8.337e-15, 7.438e-15, 5.958e-15, 5.233e-15, 3.56e-15]),
		CE(j=6, i=3, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[1.04e-14, 8.831e-15, 7.93e-15, 7.075e-15, 5.667e-15, 4.978e-15, 3.386e-15]),
		CE(j=7, i=3, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[5.062e-15, 4.299e-15, 3.86e-15, 3.443e-15, 2.758e-15, 2.423e-15, 1.648e-15]),
		CE(j=8, i=3, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[3.279e-15, 2.784e-15, 2.5e-15, 2.231e-15, 1.787e-15, 1.569e-15, 1.068e-15]),
		CE(j=9, i=3, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[1.435e-14, 8.803e-15, 6.426e-15, 4.638e-15, 2.533e-15, 1.813e-15, 7.231e-16]),
		CE(j=10, i=3, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[6.176e-16, 5.245e-16, 4.709e-16, 4.201e-16, 3.365e-16, 2.956e-16, 2.011e-16]),
		CE(j=11, i=3, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[5.998e-16, 5.093e-16, 4.573e-16, 4.08e-16, 3.268e-16, 2.871e-16, 1.953e-16]),
		CE(j=12, i=3, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[5.872e-16, 4.986e-16, 4.477e-16, 3.995e-16, 3.2e-16, 2.811e-16, 1.912e-16]),
# 		CE(j=13, i=3, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[5.858e-16, 4.975e-16, 4.467e-16, 3.985e-16, 3.192e-16, 2.804e-16, 1.907e-16]),
# 		CE(j=14, i=3, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[5.85e-16, 4.968e-16, 4.461e-16, 3.98e-16, 3.188e-16, 2.8e-16, 1.905e-16]),
# 		CE(j=15, i=3, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[5.824e-16, 4.946e-16, 4.441e-16, 3.962e-16, 3.174e-16, 2.788e-16, 1.896e-16]),
# 		CE(j=16, i=3, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[5.797e-16, 4.922e-16, 4.42e-16, 3.943e-16, 3.159e-16, 2.774e-16, 1.887e-16]),
# 		CE(j=17, i=3, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[5.733e-16, 4.869e-16, 4.372e-16, 3.9e-16, 3.124e-16, 2.744e-16, 1.867e-16]),
# 		CE(j=18, i=3, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[5.431e-16, 4.612e-16, 4.141e-16, 3.694e-16, 2.96e-16, 2.599e-16, 1.768e-16]),
# 		CE(j=19, i=3, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[5.305e-16, 4.505e-16, 4.045e-16, 3.609e-16, 2.891e-16, 2.539e-16, 1.727e-16]),
# 		CE(j=20, i=3, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[3.356e-15, 2.052e-15, 1.493e-15, 1.073e-15, 5.815e-16, 4.15e-16, 1.674e-16]),
# 		CE(j=21, i=3, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[2.065e-14, 1.263e-14, 9.187e-15, 6.605e-15, 3.578e-15, 2.554e-15, 1.03e-15]),
# 		CE(j=22, i=3, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[5.223e-16, 4.435e-16, 3.983e-16, 3.553e-16, 2.846e-16, 2.5e-16, 1.701e-16]),
# 		CE(j=23, i=3, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[5.261e-15, 3.216e-15, 2.339e-15, 1.682e-15, 9.106e-16, 6.498e-16, 2.623e-16]),
# 		CE(j=24, i=3, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[1.29e-14, 7.883e-15, 5.734e-15, 4.121e-15, 2.231e-15, 1.592e-15, 6.427e-16]),
# 		CE(j=25, i=3, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[4.791e-16, 4.068e-16, 3.653e-16, 3.259e-16, 2.611e-16, 2.293e-16, 1.56e-16]),
# 		CE(j=26, i=3, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[4.747e-16, 4.031e-16, 3.62e-16, 3.229e-16, 2.587e-16, 2.272e-16, 1.546e-16]),
# 		CE(j=27, i=3, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[4.682e-16, 3.976e-16, 3.57e-16, 3.185e-16, 2.551e-16, 2.241e-16, 1.524e-16]),
# 		CE(j=28, i=3, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[4.539e-16, 3.855e-16, 3.461e-16, 3.088e-16, 2.474e-16, 2.173e-16, 1.478e-16]),
# 		CE(j=29, i=3, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[4.471e-16, 3.797e-16, 3.409e-16, 3.042e-16, 2.437e-16, 2.14e-16, 1.456e-16]),
# 		CE(j=30, i=3, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[4.427e-16, 3.76e-16, 3.376e-16, 3.012e-16, 2.413e-16, 2.119e-16, 1.441e-16]),
# 		CE(j=31, i=3, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[4.424e-16, 3.757e-16, 3.373e-16, 3.009e-16, 2.411e-16, 2.117e-16, 1.44e-16]),
# 		CE(j=32, i=3, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[4.412e-16, 3.747e-16, 3.365e-16, 3.002e-16, 2.405e-16, 2.112e-16, 1.437e-16]),
		CE(j=5, i=4, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[9.338e-12, 7.929e-12, 7.12e-12, 6.352e-12, 5.088e-12, 4.469e-12, 3.04e-12]),
		CE(j=6, i=4, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[1.78e-12, 1.512e-12, 1.357e-12, 1.211e-12, 9.701e-13, 8.52e-13, 5.796e-13]),
		CE(j=7, i=4, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[2.582e-14, 2.193e-14, 1.969e-14, 1.757e-14, 1.407e-14, 1.236e-14, 8.408e-15]),
		CE(j=8, i=4, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[9.82e-15, 8.339e-15, 7.488e-15, 6.68e-15, 5.352e-15, 4.7e-15, 3.197e-15]),
		CE(j=9, i=4, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[1.489e-15, 1.264e-15, 1.135e-15, 1.013e-15, 8.113e-16, 7.126e-16, 4.847e-16]),
		CE(j=10, i=4, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[2.66e-14, 1.635e-14, 1.195e-14, 8.641e-15, 4.75e-15, 3.421e-15, 1.403e-15]),
		CE(j=11, i=4, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[8.277e-16, 7.028e-16, 6.311e-16, 5.63e-16, 4.51e-16, 3.961e-16, 2.695e-16]),
		CE(j=12, i=4, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[8.835e-15, 5.426e-15, 3.965e-15, 2.866e-15, 1.575e-15, 1.134e-15, 4.656e-16]),
# 		CE(j=13, i=4, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[3.055e-15, 1.876e-15, 1.371e-15, 9.911e-16, 5.444e-16, 3.92e-16, 1.61e-16]),
# 		CE(j=14, i=4, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[2.209e-15, 1.357e-15, 9.915e-16, 7.168e-16, 3.937e-16, 2.835e-16, 1.164e-16]),
# 		CE(j=15, i=4, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[7.988e-16, 6.783e-16, 6.091e-16, 5.434e-16, 4.353e-16, 3.823e-16, 2.601e-16]),
# 		CE(j=16, i=4, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[7.942e-16, 6.745e-16, 6.056e-16, 5.403e-16, 4.328e-16, 3.801e-16, 2.586e-16]),
# 		CE(j=17, i=4, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[7.838e-16, 6.656e-16, 5.976e-16, 5.332e-16, 4.271e-16, 3.751e-16, 2.552e-16]),
# 		CE(j=18, i=4, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[7.344e-16, 6.237e-16, 5.6e-16, 4.996e-16, 4.002e-16, 3.515e-16, 2.391e-16]),
# 		CE(j=19, i=4, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[7.141e-16, 6.064e-16, 5.445e-16, 4.858e-16, 3.891e-16, 3.418e-16, 2.325e-16]),
# 		CE(j=20, i=4, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[7.13e-16, 6.055e-16, 5.437e-16, 4.85e-16, 3.885e-16, 3.413e-16, 2.321e-16]),
# 		CE(j=21, i=4, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[7.114e-16, 6.041e-16, 5.424e-16, 4.839e-16, 3.876e-16, 3.405e-16, 2.316e-16]),
# 		CE(j=22, i=4, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[7.01e-16, 5.953e-16, 5.345e-16, 4.769e-16, 3.82e-16, 3.355e-16, 2.282e-16]),
# 		CE(j=23, i=4, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[6.869e-16, 5.833e-16, 5.238e-16, 4.673e-16, 3.743e-16, 3.288e-16, 2.237e-16]),
# 		CE(j=24, i=4, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[6.689e-16, 5.681e-16, 5.101e-16, 4.551e-16, 3.645e-16, 3.202e-16, 2.178e-16]),
# 		CE(j=25, i=4, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[2.244e-14, 1.376e-14, 1.004e-14, 7.246e-15, 3.967e-15, 2.853e-15, 1.179e-15]),
# 		CE(j=26, i=4, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[9.681e-15, 5.936e-15, 4.331e-15, 3.125e-15, 1.711e-15, 1.23e-15, 5.085e-16]),
# 		CE(j=27, i=4, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[6.158e-16, 5.229e-16, 4.695e-16, 4.189e-16, 3.356e-16, 2.947e-16, 2.005e-16]),
# 		CE(j=28, i=4, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[5.939e-16, 5.043e-16, 4.528e-16, 4.04e-16, 3.236e-16, 2.842e-16, 1.934e-16]),
# 		CE(j=29, i=4, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[5.834e-16, 4.954e-16, 4.449e-16, 3.969e-16, 3.179e-16, 2.792e-16, 1.9e-16]),
# 		CE(j=30, i=4, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[1.762e-14, 1.08e-14, 7.876e-15, 5.68e-15, 3.106e-15, 2.233e-15, 9.244e-16]),
# 		CE(j=31, i=4, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[1.247e-14, 7.639e-15, 5.57e-15, 4.018e-15, 2.196e-15, 1.579e-15, 6.538e-16]),
# 		CE(j=32, i=4, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[5.744e-16, 4.878e-16, 4.38e-16, 3.908e-16, 3.13e-16, 2.749e-16, 1.87e-16]),
		CE(j=6, i=5, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[3.898e-12, 3.311e-12, 2.973e-12, 2.652e-12, 2.124e-12, 1.866e-12, 1.269e-12]),
		CE(j=7, i=5, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[2.718e-14, 2.308e-14, 2.073e-14, 1.849e-14, 1.481e-14, 1.301e-14, 8.85e-15]),
		CE(j=8, i=5, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[1.011e-14, 8.581e-15, 7.705e-15, 6.874e-15, 5.507e-15, 4.837e-15, 3.29e-15]),
		CE(j=9, i=5, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[1.503e-15, 1.276e-15, 1.146e-15, 1.022e-15, 8.188e-16, 7.192e-16, 4.892e-16]),
		CE(j=10, i=5, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[4.314e-15, 2.651e-15, 1.938e-15, 1.402e-15, 7.709e-16, 5.552e-16, 2.277e-16]),
		CE(j=11, i=5, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[2.932e-14, 1.806e-14, 1.32e-14, 9.543e-15, 5.245e-15, 3.778e-15, 1.551e-15]),
		CE(j=12, i=5, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[8.119e-16, 6.895e-16, 6.191e-16, 5.523e-16, 4.424e-16, 3.886e-16, 2.644e-16]),
# 		CE(j=13, i=5, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[8.096e-16, 6.875e-16, 6.173e-16, 5.507e-16, 4.412e-16, 3.875e-16, 2.636e-16]),
# 		CE(j=14, i=5, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[3.192e-15, 1.961e-15, 1.433e-15, 1.036e-15, 5.691e-16, 4.098e-16, 1.684e-16]),
# 		CE(j=15, i=5, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[8.039e-16, 6.827e-16, 6.13e-16, 5.469e-16, 4.381e-16, 3.848e-16, 2.617e-16]),
# 		CE(j=16, i=5, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[1.346e-15, 8.264e-16, 6.038e-16, 4.365e-16, 2.398e-16, 1.727e-16, 7.098e-17]),
# 		CE(j=17, i=5, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[7.887e-16, 6.698e-16, 6.014e-16, 5.365e-16, 4.298e-16, 3.775e-16, 2.568e-16]),
# 		CE(j=18, i=5, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[7.389e-16, 6.275e-16, 5.634e-16, 5.026e-16, 4.026e-16, 3.537e-16, 2.406e-16]),
# 		CE(j=19, i=5, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[7.183e-16, 6.1e-16, 5.477e-16, 4.887e-16, 3.915e-16, 3.438e-16, 2.339e-16]),
# 		CE(j=20, i=5, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[7.173e-16, 6.091e-16, 5.469e-16, 4.879e-16, 3.909e-16, 3.433e-16, 2.335e-16]),
# 		CE(j=21, i=5, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[7.156e-16, 6.077e-16, 5.457e-16, 4.868e-16, 3.9e-16, 3.425e-16, 2.33e-16]),
# 		CE(j=22, i=5, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[7.051e-16, 5.988e-16, 5.377e-16, 4.797e-16, 3.843e-16, 3.375e-16, 2.296e-16]),
# 		CE(j=23, i=5, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[6.909e-16, 5.867e-16, 5.268e-16, 4.7e-16, 3.765e-16, 3.307e-16, 2.25e-16]),
# 		CE(j=24, i=5, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[6.728e-16, 5.713e-16, 5.13e-16, 4.577e-16, 3.666e-16, 3.22e-16, 2.191e-16]),
# 		CE(j=25, i=5, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[1.048e-14, 6.429e-15, 4.691e-15, 3.386e-15, 1.854e-15, 1.334e-15, 5.513e-16]),
# 		CE(j=26, i=5, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[1.842e-14, 1.129e-14, 8.241e-15, 5.948e-15, 3.256e-15, 2.342e-15, 9.685e-16]),
# 		CE(j=27, i=5, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[7.894e-15, 4.84e-15, 3.531e-15, 2.548e-15, 1.395e-15, 1.003e-15, 4.149e-16]),
# 		CE(j=28, i=5, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[5.97e-16, 5.07e-16, 4.552e-16, 4.061e-16, 3.253e-16, 2.858e-16, 1.944e-16]),
# 		CE(j=29, i=5, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[5.865e-16, 4.981e-16, 4.472e-16, 3.99e-16, 3.196e-16, 2.807e-16, 1.91e-16]),
# 		CE(j=30, i=5, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[5.797e-16, 4.923e-16, 4.42e-16, 3.944e-16, 3.159e-16, 2.775e-16, 1.888e-16]),
# 		CE(j=31, i=5, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[1.496e-14, 9.17e-15, 6.687e-15, 4.824e-15, 2.638e-15, 1.896e-15, 7.857e-16]),
# 		CE(j=32, i=5, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[1.08e-14, 6.62e-15, 4.828e-15, 3.482e-15, 1.904e-15, 1.369e-15, 5.672e-16]),
		CE(j=7, i=6, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[2.973e-14, 2.525e-14, 2.267e-14, 2.023e-14, 1.62e-14, 1.423e-14, 9.681e-15]),
		CE(j=8, i=6, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[1.062e-14, 9.014e-15, 8.094e-15, 7.221e-15, 5.785e-15, 5.081e-15, 3.456e-15]),
		CE(j=9, i=6, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[1.526e-15, 1.296e-15, 1.164e-15, 1.038e-15, 8.317e-16, 7.305e-16, 4.97e-16]),
		CE(j=10, i=6, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[8.731e-16, 7.414e-16, 6.657e-16, 5.939e-16, 4.758e-16, 4.179e-16, 2.843e-16]),
		CE(j=11, i=6, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[3.936e-15, 2.419e-15, 1.768e-15, 1.279e-15, 7.031e-16, 5.065e-16, 2.081e-16]),
		CE(j=12, i=6, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[9.837e-16, 6.043e-16, 4.416e-16, 3.194e-16, 1.756e-16, 1.265e-16, 5.199e-17]),
# 		CE(j=13, i=6, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[8.184e-16, 6.95e-16, 6.24e-16, 5.567e-16, 4.46e-16, 3.917e-16, 2.665e-16]),
# 		CE(j=14, i=6, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[8.171e-16, 6.939e-16, 6.23e-16, 5.558e-16, 4.453e-16, 3.911e-16, 2.66e-16]),
# 		CE(j=15, i=6, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[3.335e-14, 2.081e-14, 1.542e-14, 1.119e-14, 6.219e-15, 4.519e-15, 1.868e-15]),
# 		CE(j=16, i=6, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[3.512e-15, 2.158e-15, 1.577e-15, 1.14e-15, 6.265e-16, 4.513e-16, 1.856e-16]),
# 		CE(j=17, i=6, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[7.972e-16, 6.77e-16, 6.079e-16, 5.423e-16, 4.344e-16, 3.816e-16, 2.596e-16]),
# 		CE(j=18, i=6, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[7.465e-16, 6.339e-16, 5.692e-16, 5.078e-16, 4.068e-16, 3.573e-16, 2.431e-16]),
# 		CE(j=19, i=6, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[7.256e-16, 6.162e-16, 5.533e-16, 4.936e-16, 3.954e-16, 3.473e-16, 2.363e-16]),
# 		CE(j=20, i=6, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[7.245e-16, 6.153e-16, 5.525e-16, 4.929e-16, 3.948e-16, 3.468e-16, 2.359e-16]),
# 		CE(j=21, i=6, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[7.228e-16, 6.138e-16, 5.512e-16, 4.917e-16, 3.939e-16, 3.46e-16, 2.354e-16]),
# 		CE(j=22, i=6, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[7.122e-16, 6.048e-16, 5.431e-16, 4.845e-16, 3.881e-16, 3.409e-16, 2.319e-16]),
# 		CE(j=23, i=6, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[6.978e-16, 5.926e-16, 5.321e-16, 4.747e-16, 3.802e-16, 3.34e-16, 2.272e-16]),
# 		CE(j=24, i=6, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[6.793e-16, 5.769e-16, 5.18e-16, 4.621e-16, 3.702e-16, 3.252e-16, 2.212e-16]),
# 		CE(j=25, i=6, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[6.423e-16, 5.454e-16, 4.897e-16, 4.369e-16, 3.5e-16, 3.074e-16, 2.091e-16]),
# 		CE(j=26, i=6, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[7.784e-15, 4.774e-15, 3.484e-15, 2.515e-15, 1.377e-15, 9.91e-16, 4.101e-16]),
# 		CE(j=27, i=6, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[2.712e-14, 1.663e-14, 1.213e-14, 8.757e-15, 4.796e-15, 3.45e-15, 1.428e-15]),
# 		CE(j=28, i=6, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[6.025e-16, 5.116e-16, 4.594e-16, 4.098e-16, 3.283e-16, 2.884e-16, 1.962e-16]),
# 		CE(j=29, i=6, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[5.918e-16, 5.025e-16, 4.512e-16, 4.026e-16, 3.225e-16, 2.832e-16, 1.927e-16]),
# 		CE(j=30, i=6, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[5.849e-16, 4.967e-16, 4.46e-16, 3.979e-16, 3.187e-16, 2.799e-16, 1.904e-16]),
# 		CE(j=31, i=6, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[5.843e-16, 4.962e-16, 4.456e-16, 3.975e-16, 3.184e-16, 2.797e-16, 1.903e-16]),
# 		CE(j=32, i=6, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[1.902e-14, 1.166e-14, 8.504e-15, 6.135e-15, 3.356e-15, 2.414e-15, 1.001e-15]),
		CE(j=8, i=7, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[3.937e-14, 3.343e-14, 3.002e-14, 2.678e-14, 2.145e-14, 1.884e-14, 1.282e-14]),
		CE(j=9, i=7, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[2.091e-15, 1.775e-15, 1.594e-15, 1.422e-15, 1.139e-15, 1.001e-15, 6.807e-16]),
		CE(j=10, i=7, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[4.77e-15, 2.941e-15, 2.156e-15, 1.565e-15, 8.683e-16, 6.288e-16, 2.606e-16]),
		CE(j=11, i=7, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[1.044e-15, 8.864e-16, 7.959e-16, 7.101e-16, 5.688e-16, 4.996e-16, 3.399e-16]),
		CE(j=12, i=7, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[2.269e-14, 1.398e-14, 1.024e-14, 7.433e-15, 4.12e-15, 2.983e-15, 1.239e-15]),
# 		CE(j=13, i=7, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[1.01e-15, 8.579e-16, 7.704e-16, 6.873e-16, 5.505e-16, 4.835e-16, 3.289e-16]),
# 		CE(j=14, i=7, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[1.008e-15, 8.564e-16, 7.69e-16, 6.86e-16, 5.495e-16, 4.827e-16, 3.283e-16]),
# 		CE(j=15, i=7, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[1.002e-15, 8.51e-16, 7.642e-16, 6.817e-16, 5.461e-16, 4.797e-16, 3.263e-16]),
# 		CE(j=16, i=7, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[1.809e-15, 1.114e-15, 8.165e-16, 5.925e-16, 3.283e-16, 2.377e-16, 9.882e-17]),
# 		CE(j=17, i=7, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[2.239e-15, 1.379e-15, 1.011e-15, 7.332e-16, 4.062e-16, 2.941e-16, 1.223e-16]),
# 		CE(j=18, i=7, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[9.106e-16, 7.733e-16, 6.944e-16, 6.195e-16, 4.962e-16, 4.358e-16, 2.965e-16]),
# 		CE(j=19, i=7, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[8.82e-16, 7.49e-16, 6.726e-16, 6e-16, 4.806e-16, 4.222e-16, 2.872e-16]),
# 		CE(j=20, i=7, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[8.805e-16, 7.477e-16, 6.714e-16, 5.99e-16, 4.798e-16, 4.214e-16, 2.867e-16]),
# 		CE(j=21, i=7, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[8.782e-16, 7.458e-16, 6.697e-16, 5.974e-16, 4.786e-16, 4.203e-16, 2.859e-16]),
# 		CE(j=22, i=7, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[8.637e-16, 7.335e-16, 6.586e-16, 5.876e-16, 4.707e-16, 4.134e-16, 2.812e-16]),
# 		CE(j=23, i=7, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[8.442e-16, 7.169e-16, 6.437e-16, 5.742e-16, 4.6e-16, 4.04e-16, 2.749e-16]),
# 		CE(j=24, i=7, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[8.192e-16, 6.957e-16, 6.247e-16, 5.573e-16, 4.464e-16, 3.921e-16, 2.667e-16]),
# 		CE(j=25, i=7, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[7.695e-16, 6.535e-16, 5.868e-16, 5.235e-16, 4.193e-16, 3.683e-16, 2.505e-16]),
# 		CE(j=26, i=7, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[7.602e-16, 6.455e-16, 5.796e-16, 5.171e-16, 4.142e-16, 3.638e-16, 2.475e-16]),
# 		CE(j=27, i=7, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[7.464e-16, 6.338e-16, 5.691e-16, 5.077e-16, 4.067e-16, 3.572e-16, 2.43e-16]),
# 		CE(j=28, i=7, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[7.167e-16, 6.086e-16, 5.465e-16, 4.875e-16, 3.906e-16, 3.43e-16, 2.334e-16]),
# 		CE(j=29, i=7, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[7.026e-16, 5.967e-16, 5.358e-16, 4.78e-16, 3.829e-16, 3.363e-16, 2.288e-16]),
# 		CE(j=30, i=7, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[6.936e-16, 5.89e-16, 5.289e-16, 4.718e-16, 3.78e-16, 3.32e-16, 2.258e-16]),
# 		CE(j=31, i=7, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[6.929e-16, 5.884e-16, 5.283e-16, 4.713e-16, 3.776e-16, 3.316e-16, 2.256e-16]),
# 		CE(j=32, i=7, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[6.906e-16, 5.864e-16, 5.266e-16, 4.698e-16, 3.763e-16, 3.305e-16, 2.248e-16]),
		CE(j=9, i=8, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[2.884e-15, 2.449e-15, 2.199e-15, 1.962e-15, 1.571e-15, 1.38e-15, 9.389e-16]),
		CE(j=10, i=8, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[1.343e-15, 1.14e-15, 1.024e-15, 9.135e-16, 7.318e-16, 6.427e-16, 4.372e-16]),
		CE(j=11, i=8, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[1.282e-15, 1.089e-15, 9.777e-16, 8.722e-16, 6.987e-16, 6.137e-16, 4.175e-16]),
		CE(j=12, i=8, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[1.24e-15, 1.053e-15, 9.458e-16, 8.438e-16, 6.759e-16, 5.937e-16, 4.039e-16]),
# 		CE(j=13, i=8, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[1.236e-15, 1.049e-15, 9.423e-16, 8.407e-16, 6.734e-16, 5.915e-16, 4.024e-16]),
# 		CE(j=14, i=8, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[1.233e-15, 1.047e-15, 9.404e-16, 8.389e-16, 6.721e-16, 5.903e-16, 4.015e-16]),
# 		CE(j=15, i=8, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[1.225e-15, 1.04e-15, 9.338e-16, 8.331e-16, 6.674e-16, 5.861e-16, 3.987e-16]),
# 		CE(j=16, i=8, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[1.216e-15, 1.032e-15, 9.269e-16, 8.27e-16, 6.624e-16, 5.818e-16, 3.958e-16]),
# 		CE(j=17, i=8, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[1.195e-15, 1.015e-15, 9.112e-16, 8.13e-16, 6.512e-16, 5.72e-16, 3.891e-16]),
# 		CE(j=18, i=8, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[9.067e-15, 5.598e-15, 4.11e-15, 2.99e-15, 1.668e-15, 1.213e-15, 5.106e-16]),
# 		CE(j=19, i=8, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[1.061e-15, 9.012e-16, 8.092e-16, 7.219e-16, 5.783e-16, 5.079e-16, 3.455e-16]),
# 		CE(j=20, i=8, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[1.059e-15, 8.995e-16, 8.076e-16, 7.205e-16, 5.772e-16, 5.07e-16, 3.449e-16]),
# 		CE(j=21, i=8, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[1.056e-15, 8.969e-16, 8.053e-16, 7.184e-16, 5.755e-16, 5.055e-16, 3.439e-16]),
# 		CE(j=22, i=8, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[1.037e-15, 8.803e-16, 7.905e-16, 7.052e-16, 5.649e-16, 4.962e-16, 3.375e-16]),
# 		CE(j=23, i=8, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[1.011e-15, 8.582e-16, 7.706e-16, 6.874e-16, 5.507e-16, 4.837e-16, 3.29e-16]),
# 		CE(j=24, i=8, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[1.261e-15, 7.78e-16, 5.707e-16, 4.148e-16, 2.311e-16, 1.68e-16, 7.103e-17]),
# 		CE(j=25, i=8, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[9.119e-16, 7.744e-16, 6.953e-16, 6.203e-16, 4.969e-16, 4.365e-16, 2.969e-16]),
# 		CE(j=26, i=8, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[8.996e-16, 7.64e-16, 6.86e-16, 6.12e-16, 4.903e-16, 4.306e-16, 2.929e-16]),
# 		CE(j=27, i=8, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[8.816e-16, 7.487e-16, 6.723e-16, 5.998e-16, 4.804e-16, 4.22e-16, 2.871e-16]),
# 		CE(j=28, i=8, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[3.381e-14, 2.083e-14, 1.527e-14, 1.108e-14, 6.163e-15, 4.479e-15, 1.904e-15]),
# 		CE(j=29, i=8, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[4.879e-15, 3.006e-15, 2.202e-15, 1.599e-15, 8.887e-16, 6.459e-16, 2.747e-16]),
# 		CE(j=30, i=8, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[8.132e-16, 6.905e-16, 6.201e-16, 5.532e-16, 4.431e-16, 3.892e-16, 2.648e-16]),
# 		CE(j=31, i=8, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[8.122e-16, 6.898e-16, 6.193e-16, 5.525e-16, 4.426e-16, 3.888e-16, 2.645e-16]),
# 		CE(j=32, i=8, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[8.093e-16, 6.872e-16, 6.171e-16, 5.505e-16, 4.41e-16, 3.874e-16, 2.635e-16]),
		CE(j=10, i=9, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[7.285e-15, 6.186e-15, 5.555e-15, 4.956e-15, 3.97e-15, 3.487e-15, 2.372e-15]),
		CE(j=11, i=9, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[6.438e-15, 5.467e-15, 4.909e-15, 4.379e-15, 3.508e-15, 3.081e-15, 2.096e-15]),
		CE(j=12, i=9, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[5.911e-15, 5.02e-15, 4.507e-15, 4.021e-15, 3.221e-15, 2.829e-15, 1.925e-15]),
# 		CE(j=13, i=9, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[5.856e-15, 4.973e-15, 4.465e-15, 3.984e-15, 3.191e-15, 2.803e-15, 1.907e-15]),
# 		CE(j=14, i=9, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[5.826e-15, 4.947e-15, 4.442e-15, 3.963e-15, 3.175e-15, 2.789e-15, 1.897e-15]),
# 		CE(j=15, i=9, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[5.724e-15, 4.861e-15, 4.365e-15, 3.894e-15, 3.119e-15, 2.74e-15, 1.864e-15]),
# 		CE(j=16, i=9, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[5.62e-15, 4.772e-15, 4.285e-15, 3.823e-15, 3.062e-15, 2.69e-15, 1.83e-15]),
# 		CE(j=17, i=9, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[5.388e-15, 4.575e-15, 4.108e-15, 3.665e-15, 2.936e-15, 2.579e-15, 1.754e-15]),
# 		CE(j=18, i=9, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[4.425e-15, 3.758e-15, 3.374e-15, 3.01e-15, 2.411e-15, 2.118e-15, 1.441e-15]),
# 		CE(j=19, i=9, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[4.082e-15, 3.467e-15, 3.113e-15, 2.777e-15, 2.225e-15, 1.954e-15, 1.329e-15]),
# 		CE(j=20, i=9, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[4.065e-15, 3.452e-15, 3.1e-15, 2.765e-15, 2.215e-15, 1.946e-15, 1.324e-15]),
# 		CE(j=21, i=9, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[4.039e-15, 3.43e-15, 3.08e-15, 2.748e-15, 2.201e-15, 1.933e-15, 1.315e-15]),
# 		CE(j=22, i=9, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[3.877e-15, 3.292e-15, 2.956e-15, 2.637e-15, 2.113e-15, 1.856e-15, 1.262e-15]),
# 		CE(j=23, i=9, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[3.668e-15, 3.114e-15, 2.797e-15, 2.495e-15, 1.999e-15, 1.755e-15, 1.194e-15]),
# 		CE(j=24, i=9, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[3.416e-15, 2.901e-15, 2.605e-15, 2.324e-15, 1.862e-15, 1.635e-15, 1.112e-15]),
# 		CE(j=25, i=9, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[2.961e-15, 2.515e-15, 2.258e-15, 2.014e-15, 1.614e-15, 1.417e-15, 9.641e-16]),
# 		CE(j=26, i=9, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[2.882e-15, 2.447e-15, 2.197e-15, 1.96e-15, 1.57e-15, 1.379e-15, 9.383e-16]),
# 		CE(j=27, i=9, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[2.768e-15, 2.351e-15, 2.111e-15, 1.883e-15, 1.509e-15, 1.325e-15, 9.014e-16]),
# 		CE(j=28, i=9, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[2.537e-15, 2.154e-15, 1.934e-15, 1.726e-15, 1.382e-15, 1.214e-15, 8.26e-16]),
# 		CE(j=29, i=9, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[2.433e-15, 2.066e-15, 1.855e-15, 1.655e-15, 1.326e-15, 1.165e-15, 7.922e-16]),
# 		CE(j=30, i=9, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[2.368e-15, 2.011e-15, 1.806e-15, 1.611e-15, 1.29e-15, 1.133e-15, 7.71e-16]),
# 		CE(j=31, i=9, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[2.363e-15, 2.007e-15, 1.802e-15, 1.607e-15, 1.288e-15, 1.131e-15, 7.694e-16]),
# 		CE(j=32, i=9, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[2.347e-15, 1.993e-15, 1.789e-15, 1.596e-15, 1.279e-15, 1.123e-15, 7.641e-16]),
		CE(j=11, i=10, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[5.485e-13, 4.658e-13, 4.183e-13, 3.732e-13, 2.989e-13, 2.625e-13, 1.786e-13]),
		CE(j=12, i=10, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[2.174e-13, 1.846e-13, 1.658e-13, 1.479e-13, 1.185e-13, 1.041e-13, 7.079e-14]),
# 		CE(j=13, i=10, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[2.011e-13, 1.708e-13, 1.534e-13, 1.368e-13, 1.096e-13, 9.626e-14, 6.549e-14]),
# 		CE(j=14, i=10, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[1.929e-13, 1.638e-13, 1.471e-13, 1.312e-13, 1.051e-13, 9.231e-14, 6.28e-14]),
# 		CE(j=15, i=10, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[1.682e-13, 1.429e-13, 1.283e-13, 1.144e-13, 9.167e-14, 8.052e-14, 5.477e-14]),
# 		CE(j=16, i=10, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[1.473e-13, 1.251e-13, 1.123e-13, 1.002e-13, 8.026e-14, 7.049e-14, 4.795e-14]),
# 		CE(j=17, i=10, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[1.119e-13, 9.504e-14, 8.534e-14, 7.613e-14, 6.099e-14, 5.357e-14, 3.644e-14]),
# 		CE(j=18, i=10, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[4.344e-14, 3.689e-14, 3.313e-14, 2.955e-14, 2.367e-14, 2.079e-14, 1.414e-14]),
# 		CE(j=19, i=10, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[3.237e-14, 2.749e-14, 2.468e-14, 2.202e-14, 1.764e-14, 1.549e-14, 1.054e-14]),
# 		CE(j=20, i=10, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[3.191e-14, 2.71e-14, 2.433e-14, 2.171e-14, 1.739e-14, 1.527e-14, 1.039e-14]),
# 		CE(j=21, i=10, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[3.122e-14, 2.651e-14, 2.38e-14, 2.124e-14, 1.701e-14, 1.494e-14, 1.016e-14]),
# 		CE(j=22, i=10, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[2.728e-14, 2.317e-14, 2.08e-14, 1.856e-14, 1.487e-14, 1.306e-14, 8.883e-15]),
# 		CE(j=23, i=10, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[2.299e-14, 1.952e-14, 1.753e-14, 1.564e-14, 1.253e-14, 1.1e-14, 7.485e-15]),
# 		CE(j=24, i=10, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[1.876e-14, 1.593e-14, 1.43e-14, 1.276e-14, 1.022e-14, 8.979e-15, 6.108e-15]),
# 		CE(j=25, i=10, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[1.298e-14, 1.103e-14, 9.901e-15, 8.833e-15, 7.076e-15, 6.215e-15, 4.228e-15]),
# 		CE(j=26, i=10, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[1.217e-14, 1.034e-14, 9.281e-15, 8.28e-15, 6.633e-15, 5.826e-15, 3.963e-15]),
# 		CE(j=27, i=10, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[1.109e-14, 9.416e-15, 8.455e-15, 7.543e-15, 6.042e-15, 5.307e-15, 3.61e-15]),
# 		CE(j=28, i=10, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[9.14e-15, 7.762e-15, 6.969e-15, 6.218e-15, 4.981e-15, 4.375e-15, 2.976e-15]),
# 		CE(j=29, i=10, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[8.367e-15, 7.105e-15, 6.38e-15, 5.692e-15, 4.56e-15, 4.005e-15, 2.724e-15]),
# 		CE(j=30, i=10, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[7.912e-15, 6.719e-15, 6.033e-15, 5.382e-15, 4.312e-15, 3.787e-15, 2.576e-15]),
# 		CE(j=31, i=10, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[7.877e-15, 6.689e-15, 6.006e-15, 5.358e-15, 4.292e-15, 3.77e-15, 2.565e-15]),
# 		CE(j=32, i=10, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[7.767e-15, 6.596e-15, 5.923e-15, 5.284e-15, 4.233e-15, 3.718e-15, 2.529e-15]),
		CE(j=12, i=11, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[9.207e-13, 7.819e-13, 7.021e-13, 6.263e-13, 5.017e-13, 4.407e-13, 2.998e-13]),
# 		CE(j=13, i=11, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[7.702e-13, 6.541e-13, 5.873e-13, 5.24e-13, 4.197e-13, 3.686e-13, 2.508e-13]),
# 		CE(j=14, i=11, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[7.027e-13, 5.967e-13, 5.358e-13, 4.78e-13, 3.829e-13, 3.363e-13, 2.288e-13]),
# 		CE(j=15, i=11, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[5.298e-13, 4.499e-13, 4.04e-13, 3.604e-13, 2.887e-13, 2.536e-13, 1.725e-13]),
# 		CE(j=16, i=11, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[4.11e-13, 3.49e-13, 3.134e-13, 2.796e-13, 2.24e-13, 1.967e-13, 1.338e-13]),
# 		CE(j=17, i=11, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[2.555e-13, 2.17e-13, 1.949e-13, 1.738e-13, 1.393e-13, 1.223e-13, 8.32e-14]),
# 		CE(j=18, i=11, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[6.61e-14, 5.613e-14, 5.04e-14, 4.496e-14, 3.602e-14, 3.164e-14, 2.152e-14]),
# 		CE(j=19, i=11, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[4.57e-14, 3.881e-14, 3.485e-14, 3.109e-14, 2.49e-14, 2.187e-14, 1.488e-14]),
# 		CE(j=20, i=11, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[4.49e-14, 3.812e-14, 3.423e-14, 3.054e-14, 2.447e-14, 2.149e-14, 1.462e-14]),
# 		CE(j=21, i=11, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[4.371e-14, 3.712e-14, 3.333e-14, 2.973e-14, 2.382e-14, 2.092e-14, 1.423e-14]),
# 		CE(j=22, i=11, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[3.713e-14, 3.153e-14, 2.831e-14, 2.526e-14, 2.023e-14, 1.777e-14, 1.209e-14]),
# 		CE(j=23, i=11, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[3.029e-14, 2.572e-14, 2.309e-14, 2.06e-14, 1.651e-14, 1.45e-14, 9.862e-15]),
# 		CE(j=24, i=11, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[2.389e-14, 2.029e-14, 1.822e-14, 1.625e-14, 1.302e-14, 1.144e-14, 7.779e-15]),
# 		CE(j=25, i=11, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[1.572e-14, 1.335e-14, 1.199e-14, 1.07e-14, 8.569e-15, 7.526e-15, 5.12e-15]),
# 		CE(j=26, i=11, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[1.463e-14, 1.242e-14, 1.115e-14, 9.951e-15, 7.971e-15, 7.001e-15, 4.763e-15]),
# 		CE(j=27, i=11, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[1.319e-14, 1.12e-14, 1.006e-14, 8.971e-15, 7.186e-15, 6.312e-15, 4.294e-15]),
# 		CE(j=28, i=11, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[1.066e-14, 9.051e-15, 8.127e-15, 7.25e-15, 5.808e-15, 5.101e-15, 3.47e-15]),
# 		CE(j=29, i=11, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[9.677e-15, 8.218e-15, 7.379e-15, 6.583e-15, 5.273e-15, 4.632e-15, 3.151e-15]),
# 		CE(j=30, i=11, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[9.105e-15, 7.732e-15, 6.943e-15, 6.194e-15, 4.962e-15, 4.358e-15, 2.965e-15]),
# 		CE(j=31, i=11, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[9.061e-15, 7.695e-15, 6.909e-15, 6.164e-15, 4.938e-15, 4.337e-15, 2.95e-15]),
# 		CE(j=32, i=11, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[8.924e-15, 7.578e-15, 6.805e-15, 6.071e-15, 4.863e-15, 4.271e-15, 2.906e-15]),
# 		CE(j=13, i=12, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[3.638e-11, 3.089e-11, 2.774e-11, 2.474e-11, 1.982e-11, 1.741e-11, 1.184e-11]),
# 		CE(j=14, i=12, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[1.728e-11, 1.468e-11, 1.318e-11, 1.176e-11, 9.419e-12, 8.273e-12, 5.628e-12]),
# 		CE(j=15, i=12, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[4.487e-12, 3.811e-12, 3.422e-12, 3.053e-12, 2.445e-12, 2.148e-12, 1.461e-12]),
# 		CE(j=16, i=12, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[2.077e-12, 1.764e-12, 1.584e-12, 1.413e-12, 1.132e-12, 9.941e-13, 6.763e-13]),
# 		CE(j=17, i=12, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[7.338e-13, 6.231e-13, 5.595e-13, 4.992e-13, 3.999e-13, 3.512e-13, 2.389e-13]),
# 		CE(j=18, i=12, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[9.789e-14, 8.313e-14, 7.465e-14, 6.659e-14, 5.335e-14, 4.685e-14, 3.187e-14]),
# 		CE(j=19, i=12, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[6.217e-14, 5.279e-14, 4.74e-14, 4.229e-14, 3.388e-14, 2.976e-14, 2.024e-14]),
# 		CE(j=20, i=12, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[6.086e-14, 5.168e-14, 4.64e-14, 4.14e-14, 3.316e-14, 2.913e-14, 1.981e-14]),
# 		CE(j=21, i=12, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[5.893e-14, 5.005e-14, 4.494e-14, 4.009e-14, 3.212e-14, 2.821e-14, 1.919e-14]),
# 		CE(j=22, i=12, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[4.858e-14, 4.126e-14, 3.705e-14, 3.305e-14, 2.648e-14, 2.325e-14, 1.582e-14]),
# 		CE(j=23, i=12, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[3.835e-14, 3.256e-14, 2.924e-14, 2.609e-14, 2.09e-14, 1.835e-14, 1.249e-14]),
# 		CE(j=24, i=12, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[2.927e-14, 2.485e-14, 2.232e-14, 1.991e-14, 1.595e-14, 1.401e-14, 9.529e-15]),
# 		CE(j=25, i=12, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[1.838e-14, 1.561e-14, 1.401e-14, 1.25e-14, 1.002e-14, 8.797e-15, 5.984e-15]),
# 		CE(j=26, i=12, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[1.698e-14, 1.442e-14, 1.295e-14, 1.155e-14, 9.254e-15, 8.128e-15, 5.529e-15]),
# 		CE(j=27, i=12, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[1.517e-14, 1.288e-14, 1.156e-14, 1.032e-14, 8.265e-15, 7.259e-15, 4.938e-15]),
# 		CE(j=28, i=12, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[1.205e-14, 1.023e-14, 9.186e-15, 8.196e-15, 6.565e-15, 5.766e-15, 3.923e-15]),
# 		CE(j=29, i=12, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[1.086e-14, 9.224e-15, 8.282e-15, 7.389e-15, 5.919e-15, 5.199e-15, 3.536e-15]),
# 		CE(j=30, i=12, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[1.018e-14, 8.642e-15, 7.76e-15, 6.923e-15, 5.546e-15, 4.871e-15, 3.313e-15]),
# 		CE(j=31, i=12, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[1.012e-14, 8.597e-15, 7.72e-15, 6.887e-15, 5.517e-15, 4.846e-15, 3.296e-15]),
# 		CE(j=32, i=12, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[9.961e-15, 8.458e-15, 7.595e-15, 6.776e-15, 5.428e-15, 4.767e-15, 3.243e-15]),
# 		CE(j=14, i=13, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[9.716e-11, 8.251e-11, 7.409e-11, 6.61e-11, 5.295e-11, 4.651e-11, 3.164e-11]),
# 		CE(j=15, i=13, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[7.936e-12, 6.739e-12, 6.051e-12, 5.398e-12, 4.324e-12, 3.798e-12, 2.584e-12]),
# 		CE(j=16, i=13, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[2.91e-12, 2.472e-12, 2.219e-12, 1.98e-12, 1.586e-12, 1.393e-12, 9.476e-13]),
# 		CE(j=17, i=13, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[8.725e-13, 7.409e-13, 6.653e-13, 5.935e-13, 4.755e-13, 4.176e-13, 2.841e-13]),
# 		CE(j=18, i=13, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[1.029e-13, 8.742e-14, 7.85e-14, 7.003e-14, 5.61e-14, 4.927e-14, 3.352e-14]),
# 		CE(j=19, i=13, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[6.459e-14, 5.485e-14, 4.925e-14, 4.394e-14, 3.52e-14, 3.092e-14, 2.103e-14]),
# 		CE(j=20, i=13, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[6.32e-14, 5.367e-14, 4.819e-14, 4.299e-14, 3.444e-14, 3.025e-14, 2.058e-14]),
# 		CE(j=21, i=13, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[6.116e-14, 5.194e-14, 4.664e-14, 4.161e-14, 3.333e-14, 2.927e-14, 1.991e-14]),
# 		CE(j=22, i=13, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[5.021e-14, 4.264e-14, 3.829e-14, 3.416e-14, 2.736e-14, 2.403e-14, 1.635e-14]),
# 		CE(j=23, i=13, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[3.946e-14, 3.351e-14, 3.009e-14, 2.684e-14, 2.15e-14, 1.889e-14, 1.285e-14]),
# 		CE(j=24, i=13, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[2.999e-14, 2.547e-14, 2.287e-14, 2.04e-14, 1.634e-14, 1.435e-14, 9.764e-15]),
# 		CE(j=25, i=13, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[1.872e-14, 1.59e-14, 1.428e-14, 1.274e-14, 1.02e-14, 8.961e-15, 6.096e-15]),
# 		CE(j=26, i=13, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[1.728e-14, 1.468e-14, 1.318e-14, 1.176e-14, 9.418e-15, 8.272e-15, 5.627e-15]),
# 		CE(j=27, i=13, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[1.542e-14, 1.309e-14, 1.176e-14, 1.049e-14, 8.402e-15, 7.379e-15, 5.02e-15]),
# 		CE(j=28, i=13, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[1.222e-14, 1.038e-14, 9.319e-15, 8.314e-15, 6.66e-15, 5.849e-15, 3.979e-15]),
# 		CE(j=29, i=13, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[1.101e-14, 9.349e-15, 8.394e-15, 7.489e-15, 5.999e-15, 5.269e-15, 3.584e-15]),
# 		CE(j=30, i=13, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[1.031e-14, 8.754e-15, 7.861e-15, 7.013e-15, 5.618e-15, 4.934e-15, 3.357e-15]),
# 		CE(j=31, i=13, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[1.026e-14, 8.709e-15, 7.82e-15, 6.976e-15, 5.589e-15, 4.909e-15, 3.339e-15]),
# 		CE(j=32, i=13, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[1.009e-14, 8.567e-15, 7.693e-15, 6.863e-15, 5.498e-15, 4.829e-15, 3.285e-15]),
# 		CE(j=15, i=14, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[1.218e-11, 1.034e-11, 9.288e-12, 8.286e-12, 6.638e-12, 5.83e-12, 3.966e-12]),
# 		CE(j=16, i=14, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[3.635e-12, 3.087e-12, 2.772e-12, 2.473e-12, 1.981e-12, 1.74e-12, 1.183e-12]),
# 		CE(j=17, i=14, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[9.689e-13, 8.228e-13, 7.388e-13, 6.591e-13, 5.28e-13, 4.638e-13, 3.155e-13]),
# 		CE(j=18, i=14, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[1.059e-13, 8.997e-14, 8.079e-14, 7.207e-14, 5.774e-14, 5.071e-14, 3.45e-14]),
# 		CE(j=19, i=14, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[6.601e-14, 5.606e-14, 5.034e-14, 4.491e-14, 3.597e-14, 3.16e-14, 2.149e-14]),
# 		CE(j=20, i=14, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[6.457e-14, 5.483e-14, 4.924e-14, 4.392e-14, 3.519e-14, 3.091e-14, 2.102e-14]),
# 		CE(j=21, i=14, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[6.246e-14, 5.304e-14, 4.763e-14, 4.249e-14, 3.404e-14, 2.99e-14, 2.034e-14]),
# 		CE(j=22, i=14, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[5.116e-14, 4.345e-14, 3.901e-14, 3.48e-14, 2.788e-14, 2.449e-14, 1.666e-14]),
# 		CE(j=23, i=14, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[4.01e-14, 3.406e-14, 3.058e-14, 2.728e-14, 2.185e-14, 1.92e-14, 1.306e-14]),
# 		CE(j=24, i=14, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[3.04e-14, 2.582e-14, 2.318e-14, 2.068e-14, 1.657e-14, 1.455e-14, 9.899e-15]),
# 		CE(j=25, i=14, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[1.892e-14, 1.606e-14, 1.442e-14, 1.287e-14, 1.031e-14, 9.054e-15, 6.159e-15]),
# 		CE(j=26, i=14, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[1.745e-14, 1.482e-14, 1.331e-14, 1.187e-14, 9.511e-15, 8.354e-15, 5.683e-15]),
# 		CE(j=27, i=14, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[1.556e-14, 1.321e-14, 1.187e-14, 1.059e-14, 8.48e-15, 7.448e-15, 5.067e-15]),
# 		CE(j=28, i=14, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[1.232e-14, 1.046e-14, 9.394e-15, 8.381e-15, 6.714e-15, 5.897e-15, 4.011e-15]),
# 		CE(j=29, i=14, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[1.109e-14, 9.419e-15, 8.458e-15, 7.546e-15, 6.045e-15, 5.309e-15, 3.612e-15]),
# 		CE(j=30, i=14, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[1.038e-14, 8.818e-15, 7.918e-15, 7.064e-15, 5.659e-15, 4.97e-15, 3.381e-15]),
# 		CE(j=31, i=14, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[1.033e-14, 8.772e-15, 7.877e-15, 7.027e-15, 5.629e-15, 4.944e-15, 3.363e-15]),
# 		CE(j=32, i=14, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[1.016e-14, 8.629e-15, 7.748e-15, 6.912e-15, 5.537e-15, 4.863e-15, 3.308e-15]),
# 		CE(j=16, i=15, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[1.115e-11, 9.468e-12, 8.501e-12, 7.584e-12, 6.076e-12, 5.336e-12, 3.63e-12]),
# 		CE(j=17, i=15, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[1.476e-12, 1.253e-12, 1.125e-12, 1.004e-12, 8.043e-13, 7.064e-13, 4.806e-13]),
# 		CE(j=18, i=15, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[1.174e-13, 9.971e-14, 8.953e-14, 7.988e-14, 6.399e-14, 5.62e-14, 3.823e-14]),
# 		CE(j=19, i=15, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[7.129e-14, 6.054e-14, 5.436e-14, 4.85e-14, 3.885e-14, 3.412e-14, 2.321e-14]),
# 		CE(j=20, i=15, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[6.966e-14, 5.916e-14, 5.312e-14, 4.739e-14, 3.796e-14, 3.334e-14, 2.268e-14]),
# 		CE(j=21, i=15, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[6.729e-14, 5.714e-14, 5.131e-14, 4.577e-14, 3.667e-14, 3.221e-14, 2.191e-14]),
# 		CE(j=22, i=15, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[5.465e-14, 4.641e-14, 4.167e-14, 3.717e-14, 2.978e-14, 2.616e-14, 1.779e-14]),
# 		CE(j=23, i=15, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[4.245e-14, 3.605e-14, 3.237e-14, 2.888e-14, 2.313e-14, 2.032e-14, 1.382e-14]),
# 		CE(j=24, i=15, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[3.19e-14, 2.709e-14, 2.433e-14, 2.17e-14, 1.738e-14, 1.527e-14, 1.039e-14]),
# 		CE(j=25, i=15, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[1.961e-14, 1.666e-14, 1.496e-14, 1.334e-14, 1.069e-14, 9.387e-15, 6.386e-15]),
# 		CE(j=26, i=15, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[1.807e-14, 1.534e-14, 1.377e-14, 1.229e-14, 9.844e-15, 8.646e-15, 5.882e-15]),
# 		CE(j=27, i=15, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[1.607e-14, 1.365e-14, 1.225e-14, 1.093e-14, 8.757e-15, 7.691e-15, 5.232e-15]),
# 		CE(j=28, i=15, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[1.267e-14, 1.076e-14, 9.66e-15, 8.618e-15, 6.904e-15, 6.064e-15, 4.125e-15]),
# 		CE(j=29, i=15, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[1.139e-14, 9.67e-15, 8.683e-15, 7.746e-15, 6.205e-15, 5.45e-15, 3.708e-15]),
# 		CE(j=30, i=15, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[1.065e-14, 9.043e-15, 8.12e-15, 7.244e-15, 5.803e-15, 5.097e-15, 3.467e-15]),
# 		CE(j=31, i=15, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[1.059e-14, 8.995e-15, 8.077e-15, 7.206e-15, 5.772e-15, 5.07e-15, 3.449e-15]),
# 		CE(j=32, i=15, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[1.042e-14, 8.846e-15, 7.943e-15, 7.086e-15, 5.677e-15, 4.986e-15, 3.392e-15]),
# 		CE(j=17, i=16, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[2.688e-12, 2.283e-12, 2.05e-12, 1.829e-12, 1.465e-12, 1.287e-12, 8.752e-13]),
# 		CE(j=18, i=16, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[1.318e-13, 1.119e-13, 1.005e-13, 8.967e-14, 7.183e-14, 6.309e-14, 4.292e-14]),
# 		CE(j=19, i=16, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[7.763e-14, 6.592e-14, 5.92e-14, 5.281e-14, 4.23e-14, 3.716e-14, 2.528e-14]),
# 		CE(j=20, i=16, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[7.577e-14, 6.434e-14, 5.777e-14, 5.154e-14, 4.129e-14, 3.626e-14, 2.467e-14]),
# 		CE(j=21, i=16, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[7.305e-14, 6.203e-14, 5.57e-14, 4.969e-14, 3.981e-14, 3.496e-14, 2.379e-14]),
# 		CE(j=22, i=16, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[5.875e-14, 4.989e-14, 4.48e-14, 3.997e-14, 3.202e-14, 2.812e-14, 1.913e-14]),
# 		CE(j=23, i=16, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[4.517e-14, 3.836e-14, 3.444e-14, 3.073e-14, 2.462e-14, 2.162e-14, 1.471e-14]),
# 		CE(j=24, i=16, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[3.361e-14, 2.854e-14, 2.563e-14, 2.287e-14, 1.832e-14, 1.609e-14, 1.094e-14]),
# 		CE(j=25, i=16, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[2.039e-14, 1.732e-14, 1.555e-14, 1.387e-14, 1.111e-14, 9.76e-15, 6.64e-15]),
# 		CE(j=26, i=16, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[1.875e-14, 1.592e-14, 1.43e-14, 1.275e-14, 1.022e-14, 8.973e-15, 6.104e-15]),
# 		CE(j=27, i=16, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[1.663e-14, 1.413e-14, 1.268e-14, 1.132e-14, 9.065e-15, 7.962e-15, 5.416e-15]),
# 		CE(j=28, i=16, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[1.305e-14, 1.109e-14, 9.954e-15, 8.88e-15, 7.114e-15, 6.248e-15, 4.25e-15]),
# 		CE(j=29, i=16, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[1.171e-14, 9.945e-15, 8.93e-15, 7.967e-15, 6.382e-15, 5.605e-15, 3.813e-15]),
# 		CE(j=30, i=16, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[1.094e-14, 9.291e-15, 8.342e-15, 7.442e-15, 5.962e-15, 5.236e-15, 3.562e-15]),
# 		CE(j=31, i=16, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[1.088e-14, 9.241e-15, 8.297e-15, 7.402e-15, 5.93e-15, 5.208e-15, 3.543e-15]),
# 		CE(j=32, i=16, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[1.07e-14, 9.085e-15, 8.158e-15, 7.278e-15, 5.83e-15, 5.121e-15, 3.483e-15]),
# 		CE(j=18, i=17, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[1.789e-13, 1.519e-13, 1.364e-13, 1.217e-13, 9.748e-14, 8.562e-14, 5.824e-14]),
# 		CE(j=19, i=17, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[9.646e-14, 8.191e-14, 7.355e-14, 6.562e-14, 5.257e-14, 4.617e-14, 3.141e-14]),
# 		CE(j=20, i=17, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[9.383e-14, 7.968e-14, 7.155e-14, 6.383e-14, 5.113e-14, 4.491e-14, 3.055e-14]),
# 		CE(j=21, i=17, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[9.003e-14, 7.645e-14, 6.865e-14, 6.124e-14, 4.906e-14, 4.309e-14, 2.931e-14]),
# 		CE(j=22, i=17, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[7.049e-14, 5.986e-14, 5.375e-14, 4.795e-14, 3.841e-14, 3.374e-14, 2.295e-14]),
# 		CE(j=23, i=17, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[5.272e-14, 4.477e-14, 4.02e-14, 3.586e-14, 2.873e-14, 2.523e-14, 1.716e-14]),
# 		CE(j=24, i=17, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[3.822e-14, 3.246e-14, 2.915e-14, 2.6e-14, 2.083e-14, 1.83e-14, 1.245e-14]),
# 		CE(j=25, i=17, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[2.241e-14, 1.903e-14, 1.709e-14, 1.525e-14, 1.221e-14, 1.073e-14, 7.298e-15]),
# 		CE(j=26, i=17, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[2.051e-14, 1.742e-14, 1.564e-14, 1.395e-14, 1.118e-14, 9.816e-15, 6.678e-15]),
# 		CE(j=27, i=17, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[1.808e-14, 1.536e-14, 1.379e-14, 1.23e-14, 9.854e-15, 8.655e-15, 5.887e-15]),
# 		CE(j=28, i=17, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[1.403e-14, 1.191e-14, 1.07e-14, 9.543e-15, 7.645e-15, 6.715e-15, 4.568e-15]),
# 		CE(j=29, i=17, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[1.253e-14, 1.064e-14, 9.553e-15, 8.522e-15, 6.827e-15, 5.996e-15, 4.079e-15]),
# 		CE(j=30, i=17, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[1.167e-14, 9.911e-15, 8.9e-15, 7.94e-15, 6.36e-15, 5.586e-15, 3.8e-15]),
# 		CE(j=31, i=17, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[1.161e-14, 9.856e-15, 8.85e-15, 7.895e-15, 6.325e-15, 5.555e-15, 3.779e-15]),
# 		CE(j=32, i=17, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[1.14e-14, 9.684e-15, 8.695e-15, 7.757e-15, 6.214e-15, 5.458e-15, 3.713e-15]),
# 		CE(j=19, i=18, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[6.991e-13, 5.937e-13, 5.331e-13, 4.756e-13, 3.81e-13, 3.346e-13, 2.276e-13]),
# 		CE(j=20, i=18, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[6.4e-13, 5.435e-13, 4.88e-13, 4.354e-13, 3.488e-13, 3.063e-13, 2.084e-13]),
# 		CE(j=21, i=18, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[5.64e-13, 4.789e-13, 4.3e-13, 3.836e-13, 3.073e-13, 2.699e-13, 1.836e-13]),
# 		CE(j=22, i=18, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[2.961e-13, 2.515e-13, 2.258e-13, 2.014e-13, 1.614e-13, 1.417e-13, 9.642e-14]),
# 		CE(j=23, i=18, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[1.598e-13, 1.357e-13, 1.219e-13, 1.087e-13, 8.71e-14, 7.65e-14, 5.204e-14]),
# 		CE(j=24, i=18, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[8.993e-14, 7.637e-14, 6.858e-14, 6.118e-14, 4.901e-14, 4.305e-14, 2.928e-14]),
# 		CE(j=25, i=18, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[3.989e-14, 3.387e-14, 3.042e-14, 2.714e-14, 2.174e-14, 1.909e-14, 1.299e-14]),
# 		CE(j=26, i=18, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[3.524e-14, 2.993e-14, 2.687e-14, 2.397e-14, 1.921e-14, 1.687e-14, 1.147e-14]),
# 		CE(j=27, i=18, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[2.969e-14, 2.521e-14, 2.264e-14, 2.02e-14, 1.618e-14, 1.421e-14, 9.667e-15]),
# 		CE(j=28, i=18, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[2.128e-14, 1.808e-14, 1.623e-14, 1.448e-14, 1.16e-14, 1.019e-14, 6.93e-15]),
# 		CE(j=29, i=18, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[1.844e-14, 1.566e-14, 1.406e-14, 1.254e-14, 1.005e-14, 8.824e-15, 6.003e-15]),
# 		CE(j=30, i=18, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[1.687e-14, 1.433e-14, 1.287e-14, 1.148e-14, 9.194e-15, 8.076e-15, 5.494e-15]),
# 		CE(j=31, i=18, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[1.676e-14, 1.423e-14, 1.278e-14, 1.14e-14, 9.131e-15, 8.02e-15, 5.455e-15]),
# 		CE(j=32, i=18, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[1.639e-14, 1.392e-14, 1.25e-14, 1.115e-14, 8.932e-15, 7.846e-15, 5.337e-15]),
# 		CE(j=20, i=19, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[9.438e-11, 8.014e-11, 7.196e-11, 6.42e-11, 5.143e-11, 4.517e-11, 3.073e-11]),
# 		CE(j=21, i=19, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[1.987e-11, 1.687e-11, 1.515e-11, 1.351e-11, 1.083e-11, 9.509e-12, 6.468e-12]),
# 		CE(j=22, i=19, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[1.379e-12, 1.171e-12, 1.051e-12, 9.379e-13, 7.513e-13, 6.599e-13, 4.489e-13]),
# 		CE(j=23, i=19, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[3.939e-13, 3.345e-13, 3.004e-13, 2.68e-13, 2.147e-13, 1.885e-13, 1.283e-13]),
# 		CE(j=24, i=19, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[1.618e-13, 1.374e-13, 1.234e-13, 1.101e-13, 8.818e-14, 7.745e-14, 5.269e-14]),
# 		CE(j=25, i=19, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[5.589e-14, 4.746e-14, 4.261e-14, 3.802e-14, 3.046e-14, 2.675e-14, 1.82e-14]),
# 		CE(j=26, i=19, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[4.809e-14, 4.084e-14, 3.667e-14, 3.272e-14, 2.621e-14, 2.302e-14, 1.566e-14]),
# 		CE(j=27, i=19, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[3.921e-14, 3.329e-14, 2.99e-14, 2.667e-14, 2.137e-14, 1.877e-14, 1.277e-14]),
# 		CE(j=28, i=19, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[2.664e-14, 2.263e-14, 2.032e-14, 1.813e-14, 1.452e-14, 1.275e-14, 8.675e-15]),
# 		CE(j=29, i=19, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[2.263e-14, 1.922e-14, 1.726e-14, 1.54e-14, 1.233e-14, 1.083e-14, 7.369e-15]),
# 		CE(j=30, i=19, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[2.048e-14, 1.739e-14, 1.562e-14, 1.393e-14, 1.116e-14, 9.803e-15, 6.669e-15]),
# 		CE(j=31, i=19, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[2.032e-14, 1.726e-14, 1.55e-14, 1.382e-14, 1.107e-14, 9.727e-15, 6.617e-15]),
# 		CE(j=32, i=19, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[1.983e-14, 1.684e-14, 1.512e-14, 1.349e-14, 1.081e-14, 9.49e-15, 6.456e-15]),
# 		CE(j=21, i=20, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[4.628e-11, 3.93e-11, 3.529e-11, 3.148e-11, 2.522e-11, 2.215e-11, 1.507e-11]),
# 		CE(j=22, i=20, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[1.588e-12, 1.349e-12, 1.211e-12, 1.081e-12, 8.656e-13, 7.603e-13, 5.172e-13]),
# 		CE(j=23, i=20, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[4.206e-13, 3.572e-13, 3.207e-13, 2.862e-13, 2.292e-13, 2.013e-13, 1.37e-13]),
# 		CE(j=24, i=20, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[1.681e-13, 1.428e-13, 1.282e-13, 1.144e-13, 9.163e-14, 8.048e-14, 5.475e-14]),
# 		CE(j=25, i=20, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[5.703e-14, 4.843e-14, 4.349e-14, 3.88e-14, 3.108e-14, 2.73e-14, 1.857e-14]),
# 		CE(j=26, i=20, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[4.899e-14, 4.16e-14, 3.736e-14, 3.333e-14, 2.67e-14, 2.345e-14, 1.595e-14]),
# 		CE(j=27, i=20, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[3.986e-14, 3.384e-14, 3.039e-14, 2.711e-14, 2.172e-14, 1.908e-14, 1.298e-14]),
# 		CE(j=28, i=20, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[2.699e-14, 2.292e-14, 2.058e-14, 1.836e-14, 1.471e-14, 1.292e-14, 8.789e-15]),
# 		CE(j=29, i=20, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[2.29e-14, 1.945e-14, 1.746e-14, 1.558e-14, 1.248e-14, 1.096e-14, 7.456e-15]),
# 		CE(j=30, i=20, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[2.071e-14, 1.759e-14, 1.579e-14, 1.409e-14, 1.129e-14, 9.912e-15, 6.743e-15]),
# 		CE(j=31, i=20, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[2.055e-14, 1.745e-14, 1.567e-14, 1.398e-14, 1.12e-14, 9.835e-15, 6.69e-15]),
# 		CE(j=32, i=20, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[2.005e-14, 1.702e-14, 1.528e-14, 1.364e-14, 1.092e-14, 9.594e-15, 6.527e-15]),
# 		CE(j=22, i=21, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[2.024e-12, 1.719e-12, 1.543e-12, 1.377e-12, 1.103e-12, 9.689e-13, 6.591e-13]),
# 		CE(j=23, i=21, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[4.675e-13, 3.97e-13, 3.565e-13, 3.18e-13, 2.548e-13, 2.238e-13, 1.522e-13]),
# 		CE(j=24, i=21, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[1.786e-13, 1.517e-13, 1.362e-13, 1.215e-13, 9.733e-14, 8.549e-14, 5.815e-14]),
# 		CE(j=25, i=21, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[5.885e-14, 4.998e-14, 4.488e-14, 4.004e-14, 3.207e-14, 2.817e-14, 1.916e-14]),
# 		CE(j=26, i=21, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[5.042e-14, 4.282e-14, 3.844e-14, 3.43e-14, 2.748e-14, 2.413e-14, 1.642e-14]),
# 		CE(j=27, i=21, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[4.088e-14, 3.471e-14, 3.117e-14, 2.781e-14, 2.228e-14, 1.957e-14, 1.331e-14]),
# 		CE(j=28, i=21, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[2.754e-14, 2.339e-14, 2.1e-14, 1.874e-14, 1.501e-14, 1.318e-14, 8.967e-15]),
# 		CE(j=29, i=21, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[2.332e-14, 1.98e-14, 1.778e-14, 1.586e-14, 1.271e-14, 1.116e-14, 7.593e-15]),
# 		CE(j=30, i=21, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[2.107e-14, 1.789e-14, 1.606e-14, 1.433e-14, 1.148e-14, 1.008e-14, 6.86e-15]),
# 		CE(j=31, i=21, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[2.09e-14, 1.775e-14, 1.594e-14, 1.422e-14, 1.139e-14, 1e-14, 6.805e-15]),
# 		CE(j=32, i=21, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[2.039e-14, 1.731e-14, 1.554e-14, 1.387e-14, 1.111e-14, 9.757e-15, 6.637e-15]),
# 		CE(j=23, i=22, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[1.161e-12, 9.856e-13, 8.85e-13, 7.895e-13, 6.324e-13, 5.555e-13, 3.779e-13]),
# 		CE(j=24, i=22, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[2.806e-13, 2.383e-13, 2.139e-13, 1.909e-13, 1.529e-13, 1.343e-13, 9.135e-14]),
# 		CE(j=25, i=22, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[7.32e-14, 6.216e-14, 5.581e-14, 4.979e-14, 3.989e-14, 3.503e-14, 2.383e-14]),
# 		CE(j=26, i=22, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[6.144e-14, 5.218e-14, 4.685e-14, 4.18e-14, 3.348e-14, 2.941e-14, 2e-14]),
# 		CE(j=27, i=22, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[4.861e-14, 4.128e-14, 3.707e-14, 3.307e-14, 2.649e-14, 2.327e-14, 1.583e-14]),
# 		CE(j=28, i=22, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[3.154e-14, 2.678e-14, 2.405e-14, 2.145e-14, 1.719e-14, 1.509e-14, 1.027e-14]),
# 		CE(j=29, i=22, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[2.635e-14, 2.238e-14, 2.009e-14, 1.793e-14, 1.436e-14, 1.261e-14, 8.58e-15]),
# 		CE(j=30, i=22, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[2.363e-14, 2.007e-14, 1.802e-14, 1.608e-14, 1.288e-14, 1.131e-14, 7.694e-15]),
# 		CE(j=31, i=22, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[2.343e-14, 1.99e-14, 1.787e-14, 1.594e-14, 1.277e-14, 1.121e-14, 7.629e-15]),
# 		CE(j=32, i=22, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[2.281e-14, 1.937e-14, 1.739e-14, 1.552e-14, 1.243e-14, 1.092e-14, 7.428e-15]),
# 		CE(j=24, i=23, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[7.203e-13, 6.117e-13, 5.492e-13, 4.9e-13, 3.925e-13, 3.447e-13, 2.345e-13]),
# 		CE(j=25, i=23, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[1.049e-13, 8.912e-14, 8.002e-14, 7.139e-14, 5.719e-14, 5.023e-14, 3.417e-14]),
# 		CE(j=26, i=23, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[8.47e-14, 7.192e-14, 6.458e-14, 5.762e-14, 4.615e-14, 4.054e-14, 2.758e-14]),
# 		CE(j=27, i=23, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[6.404e-14, 5.438e-14, 4.883e-14, 4.356e-14, 3.49e-14, 3.065e-14, 2.085e-14]),
# 		CE(j=28, i=23, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[3.886e-14, 3.3e-14, 2.963e-14, 2.644e-14, 2.118e-14, 1.86e-14, 1.265e-14]),
# 		CE(j=29, i=23, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[3.176e-14, 2.697e-14, 2.421e-14, 2.16e-14, 1.73e-14, 1.52e-14, 1.034e-14]),
# 		CE(j=30, i=23, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[2.813e-14, 2.389e-14, 2.145e-14, 1.913e-14, 1.533e-14, 1.346e-14, 9.158e-15]),
# 		CE(j=31, i=23, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[2.786e-14, 2.366e-14, 2.125e-14, 1.895e-14, 1.518e-14, 1.334e-14, 9.072e-15]),
# 		CE(j=32, i=23, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[2.705e-14, 2.297e-14, 2.063e-14, 1.84e-14, 1.474e-14, 1.295e-14, 8.807e-15]),
# 		CE(j=25, i=24, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[1.995e-13, 1.694e-13, 1.521e-13, 1.357e-13, 1.087e-13, 9.548e-14, 6.495e-14]),
# 		CE(j=26, i=24, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[1.47e-13, 1.248e-13, 1.121e-13, 9.997e-14, 8.009e-14, 7.034e-14, 4.785e-14]),
# 		CE(j=27, i=24, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[1.008e-13, 8.563e-14, 7.689e-14, 6.859e-14, 5.495e-14, 4.826e-14, 3.283e-14]),
# 		CE(j=28, i=24, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[5.379e-14, 4.568e-14, 4.101e-14, 3.659e-14, 2.931e-14, 2.574e-14, 1.751e-14]),
# 		CE(j=29, i=24, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[4.222e-14, 3.585e-14, 3.219e-14, 2.872e-14, 2.301e-14, 2.021e-14, 1.375e-14]),
# 		CE(j=30, i=24, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[3.66e-14, 3.108e-14, 2.791e-14, 2.49e-14, 1.995e-14, 1.752e-14, 1.192e-14]),
# 		CE(j=31, i=24, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[3.62e-14, 3.074e-14, 2.76e-14, 2.463e-14, 1.973e-14, 1.733e-14, 1.179e-14]),
# 		CE(j=32, i=24, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[3.497e-14, 2.97e-14, 2.667e-14, 2.379e-14, 1.906e-14, 1.674e-14, 1.139e-14]),
# 		CE(j=26, i=25, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[2.993e-12, 2.542e-12, 2.282e-12, 2.036e-12, 1.631e-12, 1.433e-12, 9.745e-13]),
# 		CE(j=27, i=25, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[6.371e-13, 5.41e-13, 4.858e-13, 4.334e-13, 3.472e-13, 3.049e-13, 2.074e-13]),
# 		CE(j=28, i=25, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[1.507e-13, 1.279e-13, 1.149e-13, 1.025e-13, 8.21e-14, 7.211e-14, 4.906e-14]),
# 		CE(j=29, i=25, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[9.871e-14, 8.383e-14, 7.527e-14, 6.715e-14, 5.379e-14, 4.725e-14, 3.214e-14]),
# 		CE(j=30, i=25, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[7.839e-14, 6.657e-14, 5.977e-14, 5.332e-14, 4.272e-14, 3.752e-14, 2.552e-14]),
# 		CE(j=31, i=25, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[7.704e-14, 6.542e-14, 5.874e-14, 5.241e-14, 4.198e-14, 3.687e-14, 2.508e-14]),
# 		CE(j=32, i=25, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[7.3e-14, 6.199e-14, 5.566e-14, 4.966e-14, 3.978e-14, 3.494e-14, 2.377e-14]),
# 		CE(j=27, i=26, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[1.495e-12, 1.27e-12, 1.14e-12, 1.017e-12, 8.147e-13, 7.156e-13, 4.868e-13]),
# 		CE(j=28, i=26, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[2.055e-13, 1.745e-13, 1.567e-13, 1.398e-13, 1.12e-13, 9.838e-14, 6.692e-14]),
# 		CE(j=29, i=26, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[1.25e-13, 1.062e-13, 9.534e-14, 8.505e-14, 6.813e-14, 5.984e-14, 4.071e-14]),
# 		CE(j=30, i=26, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[9.613e-14, 8.164e-14, 7.33e-14, 6.54e-14, 5.239e-14, 4.601e-14, 3.13e-14]),
# 		CE(j=31, i=26, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[9.427e-14, 8.006e-14, 7.188e-14, 6.413e-14, 5.137e-14, 4.512e-14, 3.069e-14]),
# 		CE(j=32, i=26, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[8.873e-14, 7.535e-14, 6.765e-14, 6.036e-14, 4.835e-14, 4.247e-14, 2.889e-14]),
# 		CE(j=28, i=27, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[3.805e-13, 3.231e-13, 2.902e-13, 2.589e-13, 2.074e-13, 1.821e-13, 1.239e-13]),
# 		CE(j=29, i=27, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[1.932e-13, 1.641e-13, 1.474e-13, 1.315e-13, 1.053e-13, 9.249e-14, 6.292e-14]),
# 		CE(j=30, i=27, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[1.385e-13, 1.176e-13, 1.056e-13, 9.42e-14, 7.546e-14, 6.628e-14, 4.509e-14]),
# 		CE(j=31, i=27, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[1.352e-13, 1.148e-13, 1.031e-13, 9.194e-14, 7.365e-14, 6.469e-14, 4.401e-14]),
# 		CE(j=32, i=27, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[1.254e-13, 1.065e-13, 9.563e-14, 8.531e-14, 6.834e-14, 6.003e-14, 4.083e-14]),
# 		CE(j=29, i=28, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[1.233e-12, 1.047e-12, 9.399e-13, 8.385e-13, 6.717e-13, 5.9e-13, 4.013e-13]),
# 		CE(j=30, i=28, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[5.255e-13, 4.462e-13, 4.007e-13, 3.574e-13, 2.863e-13, 2.515e-13, 1.711e-13]),
# 		CE(j=31, i=28, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[4.982e-13, 4.231e-13, 3.799e-13, 3.389e-13, 2.715e-13, 2.385e-13, 1.622e-13]),
# 		CE(j=32, i=28, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[4.251e-13, 3.61e-13, 3.242e-13, 2.892e-13, 2.317e-13, 2.035e-13, 1.384e-13]),
# 		CE(j=30, i=29, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[2.47e-12, 2.098e-12, 1.884e-12, 1.68e-12, 1.346e-12, 1.182e-12, 8.043e-13]),
# 		CE(j=31, i=29, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[2.168e-12, 1.841e-12, 1.653e-12, 1.475e-12, 1.181e-12, 1.038e-12, 7.058e-13]),
# 		CE(j=32, i=29, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[1.515e-12, 1.287e-12, 1.155e-12, 1.031e-12, 8.257e-13, 7.252e-13, 4.933e-13]),
# 		CE(j=31, i=30, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[1.69e-10, 1.435e-10, 1.289e-10, 1.15e-10, 9.21e-11, 8.089e-11, 5.503e-11]),
# 		CE(j=32, i=30, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[1.531e-11, 1.3e-11, 1.167e-11, 1.041e-11, 8.341e-12, 7.326e-12, 4.984e-12]),
# 		CE(j=32, i=31, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[2.424e-11, 2.059e-11, 1.848e-11, 1.649e-11, 1.321e-11, 1.16e-11, 7.893e-12]),
    ]
    )


# =============================================================================
# MISSING OR UNCLEAR INFORMATION FROM StrontiumI_LR.atom
# =============================================================================
#
# 1. SPECTRAL LINES (12 lines total, lines 30-50 in .atom file)
#    - The .atom file contains broadening parameters that need conversion to
#      Lightweaver's LineBroadening format:
#      * Line type: VOIGT vs PRD (needs mapping to LineType.CRD/PRD)
#      * Nlambda: quadrature points (needs mapping to LineQuadrature)
#      * qcore, qwing: quadrature parameters
#      * vdW approximation: BARKLEM vs UNSOLD methods
#      * van der Waals coefficients for H and He
#      * Radiative damping parameter
#      * Stark broadening parameter
#    - Unclear: How to convert these parameters to VoigtBroadening objects
#
# 2. ELECTRON EXCITATION (CE) RATES (~105 rates, lines 271-375)
#    - Only structure added, not actual data
#    - Would require manual entry of all CE(j,i,temperature,rates) entries
#    - Format: CE j i rates[7] at temperatures [3000, 5000, 7000, 10000, 20000, 30000, 100000]
#
# 3. PHOTOIONIZATION CROSS-SECTIONS (lines 56-266)
#    - .atom file has detailed wavelength-dependent cross-sections for each
#      continuum transition (alpha, Nlambda, wavelength points)
#    - Lightweaver's AtomicContinuum only accepts (j, i) parameters
#    - Unclear: Where/how to specify the cross-section data in Lightweaver
#    - Some transitions use HYDROGENIC approximation instead of EXPLICIT data
#
# 4. QUANTUM NUMBERS
#    - .atom file doesn't specify J, L, S quantum numbers for levels
#    - Lightweaver's AtomicLevel accepts optional J, L, S parameters
#    - May need to derive from term labels (e.g., "5S 1SE" -> S=0, L=0, J=0)
#
# 5. LINE QUADRATURE
#    - Need to understand Lightweaver's LineQuadrature enum values
#    - .atom file uses Nlambda values (21, 25, 101, 141)
#    - Unclear mapping between Nlambda and LineQuadrature options
#
# =============================================================================


def fe_simple():
    """
    Fe I simple atom (from lightweaver.rh_atoms.Fe_simple_atom) with continua removed.
    """
    return AtomicModel(element=Element(Z=26),
        levels=[
            AtomicLevel(E=     0.000, g=9,  label="FeI 3d6 4s2   a5DE 4", stage=0, J=Fraction(4, 1),  L=2, S=Fraction(2, 1)),
            AtomicLevel(E=   415.932, g=7,  label="FeI 3d6 4s2   a5DE 3", stage=0, J=Fraction(3, 1),  L=2, S=Fraction(2, 1)),
            AtomicLevel(E=   704.004, g=5,  label="FeI 3d6 4s2   a5DE 2", stage=0, J=Fraction(2, 1),  L=2, S=Fraction(2, 1)),
            AtomicLevel(E=   888.129, g=3,  label="FeI 3d6 4s2   a5DE 1", stage=0, J=Fraction(1, 1),  L=2, S=Fraction(2, 1)),
            AtomicLevel(E=   978.072, g=1,  label="FeI 3d6 4s2   a5DE 0", stage=0, J=Fraction(0, 1),  L=2, S=Fraction(2, 1)),
            AtomicLevel(E=  6928.266, g=35, label="FeI 3d7 4s      a5FE",  stage=0, J=Fraction(17, 1), L=3, S=Fraction(2, 1)),
            AtomicLevel(E= 17550.176, g=15, label="FeI 3d7 4s      a5PE",  stage=0, J=Fraction(7, 1),  L=1, S=Fraction(2, 1)),
            AtomicLevel(E= 19350.893, g=11, label="FeI 3d6 4s 4p z7DO 5", stage=0, J=Fraction(5, 1),  L=2, S=Fraction(3, 1)),
            AtomicLevel(E= 19562.439, g=9,  label="FeI 3d6 4s 4p z7DO 4", stage=0, J=Fraction(4, 1),  L=2, S=Fraction(3, 1)),
            AtomicLevel(E= 19757.033, g=7,  label="FeI 3d6 4s 4p z7DO 3", stage=0, J=Fraction(3, 1),  L=2, S=Fraction(3, 1)),
            AtomicLevel(E= 19912.494, g=5,  label="FeI 3d6 4s 4p z7DO 2", stage=0, J=Fraction(2, 1),  L=2, S=Fraction(3, 1)),
            AtomicLevel(E= 20019.635, g=3,  label="FeI 3d6 4s 4p z7DO 1", stage=0, J=Fraction(1, 1),  L=2, S=Fraction(3, 1)),
            AtomicLevel(E= 25899.986, g=25, label="FeI 3d6 4s 4p   z5DO",  stage=0, J=Fraction(12, 1), L=2, S=Fraction(2, 1)),
            AtomicLevel(E= 26874.549, g=35, label="FeI 3d6 4s 4p   z5FO",  stage=0, J=Fraction(17, 1), L=3, S=Fraction(2, 1)),
            AtomicLevel(E= 63737.000, g=50, label="FeII 3d6 4s     a6DE",  stage=1, J=Fraction(49, 2), L=2, S=Fraction(5, 2)),
        ],
        lines=[
            VoigtLine(j=7,  i=0, f=7.090e-06, type=LineType.CRD, quadrature=LinearCoreExpWings(qCore=2, qWing=4,    Nlambda=49), broadening=LineBroadening(natural=[RadiativeBroadening(gamma=1450)],      elastic=[VdwUnsold(vals=[1.0, 1.0]), MultiplicativeStarkBroadening(coeff=0)])),
            VoigtLine(j=9,  i=2, f=5.000e-06, type=LineType.CRD, quadrature=LinearCoreExpWings(qCore=1, qWing=4,    Nlambda=61), broadening=LineBroadening(natural=[RadiativeBroadening(gamma=392)],        elastic=[VdwUnsold(vals=[2.0, 2.0]), MultiplicativeStarkBroadening(coeff=0)])),
            VoigtLine(j=10, i=3, f=5.740e-06, type=LineType.CRD, quadrature=LinearCoreExpWings(qCore=2, qWing=4,    Nlambda=41), broadening=LineBroadening(natural=[RadiativeBroadening(gamma=832)],        elastic=[VdwUnsold(vals=[1.0, 1.0]), MultiplicativeStarkBroadening(coeff=0)])),
            VoigtLine(j=11, i=4, f=3.000e-05, type=LineType.CRD, quadrature=LinearCoreExpWings(qCore=1, qWing=4,    Nlambda=61), broadening=LineBroadening(natural=[RadiativeBroadening(gamma=930)],        elastic=[VdwUnsold(vals=[2.0, 2.0]), MultiplicativeStarkBroadening(coeff=0)])),
            VoigtLine(j=8,  i=0, f=1.930e-05, type=LineType.CRD, quadrature=LinearCoreExpWings(qCore=2, qWing=4,    Nlambda=61), broadening=LineBroadening(natural=[RadiativeBroadening(gamma=4930)],       elastic=[VdwUnsold(vals=[1.0, 1.0]), MultiplicativeStarkBroadening(coeff=0)])),
            VoigtLine(j=9,  i=1, f=1.530e-05, type=LineType.CRD, quadrature=LinearCoreExpWings(qCore=2, qWing=4,    Nlambda=53), broadening=LineBroadening(natural=[RadiativeBroadening(gamma=3830)],       elastic=[VdwUnsold(vals=[1.0, 1.0]), MultiplicativeStarkBroadening(coeff=0)])),
            VoigtLine(j=10, i=2, f=9.310e-06, type=LineType.CRD, quadrature=LinearCoreExpWings(qCore=2, qWing=4,    Nlambda=45), broadening=LineBroadening(natural=[RadiativeBroadening(gamma=2290)],       elastic=[VdwUnsold(vals=[1.0, 1.0]), MultiplicativeStarkBroadening(coeff=0)])),
            VoigtLine(j=11, i=3, f=5.420e-06, type=LineType.CRD, quadrature=LinearCoreExpWings(qCore=2, qWing=4,    Nlambda=41), broadening=LineBroadening(natural=[RadiativeBroadening(gamma=1320)],       elastic=[VdwUnsold(vals=[1.0, 1.0]), MultiplicativeStarkBroadening(coeff=0)])),
            VoigtLine(j=9,  i=0, f=3.900e-07, type=LineType.CRD, quadrature=LinearCoreExpWings(qCore=2, qWing=4,    Nlambda=41), broadening=LineBroadening(natural=[RadiativeBroadening(gamma=130)],        elastic=[VdwUnsold(vals=[1.0, 1.0]), MultiplicativeStarkBroadening(coeff=0)])),
            VoigtLine(j=10, i=1, f=1.070e-07, type=LineType.CRD, quadrature=LinearCoreExpWings(qCore=2, qWing=4,    Nlambda=41), broadening=LineBroadening(natural=[RadiativeBroadening(gamma=38)],         elastic=[VdwUnsold(vals=[1.0, 1.0]), MultiplicativeStarkBroadening(coeff=0)])),
            VoigtLine(j=12, i=0, f=6.301e-02, type=LineType.CRD, quadrature=LinearCoreExpWings(qCore=2, qWing=68.7, Nlambda=41), broadening=LineBroadening(natural=[RadiativeBroadening(gamma=1.135e+07)],  elastic=[VdwUnsold(vals=[1.0, 1.0]), MultiplicativeStarkBroadening(coeff=0)])),
            VoigtLine(j=13, i=0, f=1.243e-01, type=LineType.CRD, quadrature=LinearCoreExpWings(qCore=2, qWing=89.2, Nlambda=41), broadening=LineBroadening(natural=[RadiativeBroadening(gamma=1.204e+07)],  elastic=[VdwUnsold(vals=[1.0, 1.0]), MultiplicativeStarkBroadening(coeff=0)])),
            VoigtLine(j=7,  i=5, f=6.531e-07, type=LineType.CRD, quadrature=LinearCoreExpWings(qCore=2, qWing=4,    Nlambda=41), broadening=LineBroadening(natural=[RadiativeBroadening(gamma=271.9)],      elastic=[VdwUnsold(vals=[1.0, 1.0]), MultiplicativeStarkBroadening(coeff=0)])),
            VoigtLine(j=12, i=5, f=4.729e-03, type=LineType.CRD, quadrature=LinearCoreExpWings(qCore=2, qWing=18.5, Nlambda=41), broadening=LineBroadening(natural=[RadiativeBroadening(gamma=2.034e+06)],  elastic=[VdwUnsold(vals=[1.0, 1.0]), MultiplicativeStarkBroadening(coeff=0)])),
            VoigtLine(j=13, i=5, f=2.962e-04, type=LineType.CRD, quadrature=LinearCoreExpWings(qCore=2, qWing=7.8,  Nlambda=61), broadening=LineBroadening(natural=[RadiativeBroadening(gamma=78950)],      elastic=[VdwUnsold(vals=[1.0, 1.0]), MultiplicativeStarkBroadening(coeff=0)])),
        ],
        continua=[
            HydrogenicContinuum(j=14, i=0, NlambdaGen=15, alpha0=6.025e-22, minWavelength=52.3),
        ],
        collisions=[
            CE(j=1,  i=0,  temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[3.584e-13, 3.044e-13, 2.733e-13, 2.438e-13, 1.953e-13, 1.715e-13, 1.167e-13]),
            CE(j=2,  i=0,  temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[1.481e-13, 1.257e-13, 1.129e-13, 1.007e-13, 8.068e-14, 7.086e-14, 4.82e-14]),
            CE(j=3,  i=0,  temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[1.002e-13, 8.509e-14, 7.641e-14, 6.817e-14, 5.461e-14, 4.796e-14, 3.263e-14]),
            CE(j=4,  i=0,  temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[8.521e-14, 7.236e-14, 6.498e-14, 5.797e-14, 4.644e-14, 4.079e-14, 2.775e-14]),
            CE(j=5,  i=0,  temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[3.178e-15, 2.698e-15, 2.423e-15, 2.162e-15, 1.732e-15, 1.521e-15, 1.035e-15]),
            CE(j=6,  i=0,  temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[6.667e-16, 5.662e-16, 5.084e-16, 4.535e-16, 3.633e-16, 3.191e-16, 2.171e-16]),
            CE(j=7,  i=0,  temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[9.583e-19, 5.816e-19, 4.201e-19, 2.988e-19, 1.568e-19, 1.089e-19, 3.91e-20]),
            CE(j=8,  i=0,  temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[2.567e-18, 1.558e-18, 1.125e-18, 8.002e-19, 4.197e-19, 2.914e-19, 1.047e-19]),
            CE(j=9,  i=0,  temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[5.11e-20,  3.101e-20, 2.24e-20,  1.593e-20, 8.354e-21, 5.8e-21,   2.083e-21]),
            CE(j=10, i=0,  temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[5.393e-16, 4.579e-16, 4.112e-16, 3.668e-16, 2.939e-16, 2.581e-16, 1.756e-16]),
            CE(j=11, i=0,  temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[5.344e-16, 4.538e-16, 4.075e-16, 3.636e-16, 2.912e-16, 2.558e-16, 1.74e-16]),
            CE(j=12, i=0,  temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[5.477e-15, 3.318e-15, 2.392e-15, 1.698e-15, 8.863e-16, 6.135e-16, 2.198e-16]),
            CE(j=13, i=0,  temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[1.02e-14,  6.18e-15,  4.456e-15, 3.162e-15, 1.649e-15, 1.141e-15, 4.086e-16]),
            CE(j=2,  i=1,  temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[6.643e-13, 5.641e-13, 5.065e-13, 4.519e-13, 3.62e-13,  3.18e-13,  2.163e-13]),
            CE(j=3,  i=1,  temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[2.896e-13, 2.459e-13, 2.208e-13, 1.97e-13,  1.578e-13, 1.386e-13, 9.429e-14]),
            CE(j=4,  i=1,  temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[2.161e-13, 1.835e-13, 1.648e-13, 1.47e-13,  1.177e-13, 1.034e-13, 7.035e-14]),
            CE(j=5,  i=1,  temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[3.526e-15, 2.994e-15, 2.688e-15, 2.398e-15, 1.921e-15, 1.688e-15, 1.148e-15]),
            CE(j=6,  i=1,  temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[6.941e-16, 5.895e-16, 5.293e-16, 4.722e-16, 3.783e-16, 3.322e-16, 2.26e-16]),
            CE(j=7,  i=1,  temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[5.869e-16, 4.984e-16, 4.475e-16, 3.992e-16, 3.198e-16, 2.809e-16, 1.911e-16]),
            CE(j=8,  i=1,  temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[5.76e-16,  4.891e-16, 4.392e-16, 3.918e-16, 3.139e-16, 2.757e-16, 1.875e-16]),
            CE(j=9,  i=1,  temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[2.059e-18, 1.249e-18, 9.025e-19, 6.421e-19, 3.369e-19, 2.34e-19,  8.409e-20]),
            CE(j=10, i=1,  temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[1.423e-20, 8.634e-21, 6.236e-21, 4.436e-21, 2.327e-21, 1.616e-21, 5.809e-22]),
            CE(j=11, i=1,  temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[5.536e-16, 4.701e-16, 4.221e-16, 3.766e-16, 3.017e-16, 2.65e-16,  1.803e-16]),
            CE(j=12, i=1,  temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[3.563e-16, 3.026e-16, 2.717e-16, 2.424e-16, 1.942e-16, 1.705e-16, 1.16e-16]),
            CE(j=13, i=1,  temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[3.345e-16, 2.841e-16, 2.551e-16, 2.276e-16, 1.823e-16, 1.601e-16, 1.089e-16]),
            CE(j=3,  i=2,  temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[1.409e-12, 1.197e-12, 1.074e-12, 9.586e-13, 7.679e-13, 6.744e-13, 4.588e-13]),
            CE(j=4,  i=2,  temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[7.223e-13, 6.134e-13, 5.508e-13, 4.914e-13, 3.936e-13, 3.457e-13, 2.352e-13]),
            CE(j=5,  i=2,  temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[3.804e-15, 3.231e-15, 2.901e-15, 2.588e-15, 2.073e-15, 1.821e-15, 1.239e-15]),
            CE(j=6,  i=2,  temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[7.142e-16, 6.065e-16, 5.446e-16, 4.858e-16, 3.892e-16, 3.418e-16, 2.325e-16]),
            CE(j=7,  i=2,  temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[6.022e-16, 5.114e-16, 4.592e-16, 4.096e-16, 3.281e-16, 2.882e-16, 1.961e-16]),
            CE(j=8,  i=2,  temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[5.909e-16, 5.018e-16, 4.505e-16, 4.019e-16, 3.22e-16,  2.828e-16, 1.924e-16]),
            CE(j=9,  i=2,  temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[6.855e-19, 4.161e-19, 3.006e-19, 2.139e-19, 1.123e-19, 7.8e-20,   2.805e-20]),
            CE(j=10, i=2,  temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[1.261e-18, 7.654e-19, 5.529e-19, 3.934e-19, 2.065e-19, 1.434e-19, 5.157e-20]),
            CE(j=11, i=2,  temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[5.676e-16, 4.82e-16,  4.328e-16, 3.861e-16, 3.093e-16, 2.716e-16, 1.848e-16]),
            CE(j=12, i=2,  temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[3.632e-16, 3.084e-16, 2.769e-16, 2.47e-16,  1.979e-16, 1.738e-16, 1.182e-16]),
            CE(j=13, i=2,  temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[3.407e-16, 2.893e-16, 2.598e-16, 2.318e-16, 1.857e-16, 1.631e-16, 1.109e-16]),
            CE(j=4,  i=3,  temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[4.695e-12, 3.987e-12, 3.58e-12,  3.194e-12, 2.559e-12, 2.247e-12, 1.529e-12]),
            CE(j=5,  i=3,  temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[4.001e-15, 3.398e-15, 3.051e-15, 2.722e-15, 2.18e-15,  1.915e-15, 1.303e-15]),
            CE(j=6,  i=3,  temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[7.275e-16, 6.178e-16, 5.547e-16, 4.949e-16, 3.964e-16, 3.482e-16, 2.369e-16]),
            CE(j=7,  i=3,  temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[6.123e-16, 5.2e-16,   4.669e-16, 4.165e-16, 3.337e-16, 2.931e-16, 1.994e-16]),
            CE(j=8,  i=3,  temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[6.007e-16, 5.101e-16, 4.58e-16,  4.086e-16, 3.273e-16, 2.875e-16, 1.956e-16]),
            CE(j=9,  i=3,  temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[5.903e-16, 5.013e-16, 4.501e-16, 4.016e-16, 3.217e-16, 2.825e-16, 1.922e-16]),
            CE(j=10, i=3,  temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[7.868e-19, 4.776e-19, 3.451e-19, 2.455e-19, 1.289e-19, 8.956e-20, 3.221e-20]),
            CE(j=11, i=3,  temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[7.368e-19, 4.472e-19, 3.231e-19, 2.299e-19, 1.207e-19, 8.384e-20, 3.016e-20]),
            CE(j=12, i=3,  temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[3.677e-16, 3.122e-16, 2.803e-16, 2.501e-16, 2.004e-16, 1.76e-16,  1.197e-16]),
            CE(j=13, i=3,  temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[3.448e-16, 2.928e-16, 2.629e-16, 2.346e-16, 1.879e-16, 1.65e-16,  1.123e-16]),
            CE(j=5,  i=4,  temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[4.103e-15, 3.484e-15, 3.129e-15, 2.791e-15, 2.236e-15, 1.964e-15, 1.336e-15]),
            CE(j=6,  i=4,  temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[7.341e-16, 6.234e-16, 5.598e-16, 4.994e-16, 4.001e-16, 3.514e-16, 2.39e-16]),
            CE(j=7,  i=4,  temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[6.173e-16, 5.242e-16, 4.707e-16, 4.2e-16,   3.364e-16, 2.955e-16, 2.01e-16]),
            CE(j=8,  i=4,  temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[6.056e-16, 5.143e-16, 4.618e-16, 4.12e-16,  3.3e-16,   2.898e-16, 1.972e-16]),
            CE(j=9,  i=4,  temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[5.951e-16, 5.053e-16, 4.537e-16, 4.048e-16, 3.243e-16, 2.848e-16, 1.938e-16]),
            CE(j=10, i=4,  temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[5.869e-16, 4.984e-16, 4.475e-16, 3.992e-16, 3.198e-16, 2.809e-16, 1.911e-16]),
            CE(j=11, i=4,  temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[4.102e-18, 2.49e-18,  1.799e-18, 1.28e-18,  6.721e-19, 4.67e-19,  1.68e-19]),
            CE(j=12, i=4,  temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[3.699e-16, 3.141e-16, 2.82e-16,  2.516e-16, 2.016e-16, 1.77e-16,  1.204e-16]),
            CE(j=13, i=4,  temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[3.468e-16, 2.945e-16, 2.644e-16, 2.359e-16, 1.89e-16,  1.66e-16,  1.129e-16]),
            CE(j=6,  i=5,  temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[1.55e-15,  1.316e-15, 1.182e-15, 1.054e-15, 8.446e-16, 7.418e-16, 5.047e-16]),
            CE(j=7,  i=5,  temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[1.55e-19,  9.449e-20, 6.851e-20, 4.898e-20, 2.6e-20,   1.818e-20, 6.605e-21]),
            CE(j=8,  i=5,  temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[1.158e-15, 9.834e-16, 8.83e-16,  7.878e-16, 6.311e-16, 5.543e-16, 3.771e-16]),
            CE(j=9,  i=5,  temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[1.129e-15, 9.585e-16, 8.607e-16, 7.678e-16, 6.151e-16, 5.402e-16, 3.675e-16]),
            CE(j=10, i=5,  temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[1.106e-15, 9.393e-16, 8.434e-16, 7.524e-16, 6.028e-16, 5.294e-16, 3.601e-16]),
            CE(j=11, i=5,  temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[1.091e-15, 9.264e-16, 8.319e-16, 7.421e-16, 5.945e-16, 5.222e-16, 3.552e-16]),
            CE(j=12, i=5,  temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[5.984e-16, 3.635e-16, 2.628e-16, 1.872e-16, 9.864e-17, 6.872e-17, 2.501e-17]),
            CE(j=13, i=5,  temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[3.472e-17, 2.109e-17, 1.524e-17, 1.085e-17, 5.712e-18, 3.977e-18, 1.447e-18]),
            CE(j=7,  i=6,  temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[3.056e-14, 2.595e-14, 2.33e-14,  2.079e-14, 1.665e-14, 1.463e-14, 9.951e-15]),
            CE(j=8,  i=6,  temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[2.536e-14, 2.154e-14, 1.934e-14, 1.725e-14, 1.382e-14, 1.214e-14, 8.257e-15]),
            CE(j=9,  i=6,  temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[2.172e-14, 1.844e-14, 1.656e-14, 1.477e-14, 1.183e-14, 1.039e-14, 7.071e-15]),
            CE(j=10, i=6,  temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[1.937e-14, 1.645e-14, 1.477e-14, 1.318e-14, 1.056e-14, 9.271e-15, 6.307e-15]),
            CE(j=11, i=6,  temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[1.798e-14, 1.527e-14, 1.371e-14, 1.223e-14, 9.797e-15, 8.605e-15, 5.854e-15]),
            CE(j=12, i=6,  temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[2.322e-15, 1.972e-15, 1.771e-15, 1.58e-15,  1.266e-15, 1.112e-15, 7.561e-16]),
            CE(j=13, i=6,  temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[1.929e-15, 1.638e-15, 1.471e-15, 1.312e-15, 1.051e-15, 9.234e-16, 6.281e-16]),
            CE(j=8,  i=7,  temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[1.116e-12, 9.477e-13, 8.509e-13, 7.592e-13, 6.081e-13, 5.341e-13, 3.634e-13]),
            CE(j=9,  i=7,  temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[3.73e-13,  3.168e-13, 2.845e-13, 2.538e-13, 2.033e-13, 1.785e-13, 1.215e-13]),
            CE(j=10, i=7,  temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[2.164e-13, 1.838e-13, 1.65e-13,  1.472e-13, 1.179e-13, 1.036e-13, 7.047e-14]),
            CE(j=11, i=7,  temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[1.614e-13, 1.371e-13, 1.231e-13, 1.098e-13, 8.795e-14, 7.725e-14, 5.255e-14]),
            CE(j=12, i=7,  temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[3.493e-15, 2.966e-15, 2.663e-15, 2.376e-15, 1.903e-15, 1.672e-15, 1.137e-15]),
            CE(j=13, i=7,  temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[2.767e-15, 2.349e-15, 2.11e-15,  1.882e-15, 1.508e-15, 1.324e-15, 9.008e-16]),
            CE(j=9,  i=8,  temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[1.284e-12, 1.09e-12,  9.791e-13, 8.735e-13, 6.998e-13, 6.146e-13, 4.181e-13]),
            CE(j=10, i=8,  temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[4.788e-13, 4.066e-13, 3.651e-13, 3.257e-13, 2.609e-13, 2.292e-13, 1.559e-13]),
            CE(j=11, i=8,  temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[3.057e-13, 2.596e-13, 2.331e-13, 2.08e-13,  1.666e-13, 1.463e-13, 9.955e-14]),
            CE(j=12, i=8,  temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[3.691e-15, 3.134e-15, 2.814e-15, 2.511e-15, 2.011e-15, 1.766e-15, 1.202e-15]),
            CE(j=13, i=8,  temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[2.902e-15, 2.465e-15, 2.213e-15, 1.974e-15, 1.582e-15, 1.389e-15, 9.45e-16]),
            CE(j=10, i=9,  temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[1.872e-12, 1.59e-12,  1.428e-12, 1.274e-12, 1.02e-12,  8.962e-13, 6.097e-13]),
            CE(j=11, i=9,  temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[7.761e-13, 6.591e-13, 5.918e-13, 5.28e-13,  4.229e-13, 3.715e-13, 2.527e-13]),
            CE(j=12, i=9,  temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[3.889e-15, 3.303e-15, 2.966e-15, 2.646e-15, 2.119e-15, 1.861e-15, 1.266e-15]),
            CE(j=13, i=9,  temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[3.037e-15, 2.579e-15, 2.316e-15, 2.066e-15, 1.655e-15, 1.454e-15, 9.888e-16]),
            CE(j=11, i=10, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[3.5e-12,   2.972e-12, 2.668e-12, 2.381e-12, 1.907e-12, 1.675e-12, 1.139e-12]),
            CE(j=12, i=10, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[4.06e-15,  3.448e-15, 3.096e-15, 2.762e-15, 2.213e-15, 1.943e-15, 1.322e-15]),
            CE(j=13, i=10, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[3.152e-15, 2.676e-15, 2.403e-15, 2.144e-15, 1.717e-15, 1.508e-15, 1.026e-15]),
            CE(j=12, i=11, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[4.185e-15, 3.554e-15, 3.191e-15, 2.847e-15, 2.281e-15, 2.003e-15, 1.363e-15]),
            CE(j=13, i=11, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[3.235e-15, 2.747e-15, 2.467e-15, 2.201e-15, 1.763e-15, 1.548e-15, 1.053e-15]),
            CE(j=13, i=12, temperature=[3000.0, 5000.0, 7000.0, 10000.0, 20000.0, 30000.0, 100000.0], rates=[8.573e-14, 7.28e-14,  6.537e-14, 5.832e-14, 4.672e-14, 4.103e-14, 2.791e-14]),
        ],
    )
