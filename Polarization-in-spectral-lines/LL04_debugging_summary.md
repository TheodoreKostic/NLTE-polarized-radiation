# Summary of the debugging session

## Main objective

Implement the generalized emission profile of LL04 Eq. (10.40) and
reproduce Fig. 13.6 (frequency-dependent Stokes I,Q,U,V profiles
including Hanle and Zeeman effects).

## Generalized profile `Phi_generalized`

-   Implemented Eq. (10.40) for the special case Jl=0 -\> Ju=1.
-   Used:
    -   Ml = 0 (only lower magnetic sublevel),
    -   q = Mu,
    -   q' = Mu' from the 3-j selection rules.
-   The implementation computes
    -   angular coefficients (products of four Wigner 3-j symbols),
    -   multiplied by the averaged Zeeman profile 1/2(phi(Mu)+phi(Mu')).

### Important observations

-   The coefficients are independent of `vH`, as expected.
-   For K=2, K'=2, Q=0 only diagonal terms survive:
    -   Mu=-1,Mup=-1
    -   Mu=0,Mup=0
    -   Mu=1,Mup=1
-   All off-diagonal terms vanish because the last 3-j symbol requires
    q=q', hence Mu=Mu'.
-   Therefore Phi22(Q=0) becomes a weighted sum of Zeeman components
    rather than interference terms.

## Meaning of `vH`

The implementation currently uses

``` python
shift = (Mu-Ml)*vH
```

where

vH = nu_L / Delta_nu_D

is the Zeeman splitting expressed in Doppler widths.

This is **not** the Hanle parameter

Hu = omega_L / A_ul.

The two quantities are proportional to magnetic field but are physically
different.

## Hanle implementation

The Hanle operator is already implemented through

-   `hanle_operator()`
-   `apply_hanle()`

which returns the Hanle-modified rho\^2_Q components.

During synthesis the returned rho should be used directly, not
multiplied again by an explicit Hanle factor.

## Spectral synthesis

A frequency-dependent synthesis routine was built using

-   Phi00
-   Phi02
-   Phi20
-   Phi22

together with

-   radiation tensors
-   Hanle-modified rho
-   geometrical tensors T(i,K,Q).

## Current discrepancy

The synthesized profiles do not resemble Fig. 13.6.

Observed: - I, Q and U have incorrect amplitudes. - Q/I reaches \~1.7 at
line centre, which is physically impossible. - U is comparatively
reasonable. - V has not yet been implemented.

## Diagnostics performed

Verified: - generalized profile coefficients, - rho_Q values, -
T(i,K,Q), - dependence on vH, - generalized profile shapes.

The latest diagnostic requested is an emissivity breakdown by block:

-   (0,0)
-   (0,2)
-   (2,0)
-   (2,2)

to determine which contribution dominates the total emissivity.

A helper routine `emissivity_breakdown()` was written for this purpose.

## Current hypothesis

The remaining discrepancy is unlikely to originate from
`Phi_generalized`.

More likely candidates are: 1. normalization of emissivity blocks, 2.
normalization/convention of T(i,K,Q), 3. mixing conventions between Eq.
(10.39), Eq. (10.40), and the synthesis equation.

The next debugging step is to inspect the contribution of each
emissivity block separately.
