# LL04 Debugging Summary — Part 2

## Objective

Continue debugging the implementation of the polarized emissivity (Eq. 13.20 of LL04) in order to reproduce **Fig. 13.6**, especially

- Gaussian-like **I**
- Gaussian-like **Q**
- Gaussian-like **U**
- antisymmetric **V**

while using the generalized profiles of Eq. (10.40).

---

# 1. Stokes synthesis using generalized profiles

The synthesis was rewritten to compute

\[
I(\nu),\;
Q(\nu),\;
U(\nu),\;
V(\nu)
\]

directly from

\[
\varepsilon_i(\nu,\Omega)
=
\sum_{KK'Q}
T^{K'}_Q(i,\Omega)\,
\rho^K_Q\,
\Phi^{KK'}_Q(\nu),
\]

using

- `Phi_generalized()`
- `apply_hanle()`
- tensor `T(i,K,Q,...)`.

Initially only K=2 contributions were used.

---

# 2. Realization that Fig. 13.6 contains Stokes profiles, not profile tensors

A distinction was made between

- generalized profile tensors
  \[
  \Phi^{KK'}_Q(\nu),
  \]

and

- the emitted Stokes profiles

obtained only after contraction with

- radiation tensors,
- geometrical tensors,
- Hanle solution.

---

# 3. Confusion about using only K=2

A major point of discussion was

why Fig. 13.6 could apparently be reproduced using only

\[
K=2.
\]

Eventually we concluded:

- Eq. (13.20) sums over **all** K and K'.
- The atomic polarization lives in K=2.
- But the profile tensors exist for many (K,K') combinations.

---

# 4. Discussion of Eq. (10.40)

We revisited the implementation

```python
Phi_generalized(...)
```

and compared it carefully with Eq. (10.40).

The equation sums over

- Mu
- Mu'
- Ml
- q
- q'

whereas the code fixed

```python
Ml = 0
```

because

\[
J_l=0.
\]

This was confirmed to be correct.

---

# 5. Meaning of vH

Clarified that

```python
vH
```

is **not** the Hanle parameter.

Instead,

\[
v_H
=
\frac{\nu_L}{\Delta\nu_D},
\]

the Zeeman splitting expressed in Doppler-width units.

It appears only inside

```python
phi_transition()
```

through

```python
shift = (Mu-Ml)*vH
```

and therefore only changes the shifted Gaussian centers.

---

# 6. Why the Wigner coefficients do not depend on vH

While printing

```python
Mu
Mup
coeff
```

for different values of

```python
vH
```

the coefficients were identical.

This was expected because

Eq. (10.40) separates into

- purely angular coefficients
- spectral profiles

The angular coefficient

```python
term
```

contains only

- Wigner 3j symbols
- phase factors

and therefore is independent of magnetic splitting.

---

# 7. Investigation of missing cross terms

Printing every coefficient for

```python
K=2
Kp=2
Q=0
```

showed

only

```text
Mu=Mup
```

terms survived,

while all

```text
Mu != Mup
```

coefficients vanished.

This immediately explained why the generalized profile became simply

a weighted sum of shifted Gaussians.

This also suggested that the antisymmetric structure expected for V could not originate from these coefficients alone.

---

# 8. Construction of a diagnostic plot of Φ

A dedicated diagnostic script was written to plot

```python
Phi_generalized(...)
```

for

```text
Q=0
Q=1
Q=2
```

versus frequency.

The results showed

- Q=0 produced an antisymmetric curve
- Q=1 also antisymmetric
- Q=2 essentially vanished

These plots confirmed that the generalized profiles themselves already contain dispersive-looking structure.

---

# 9. Discovery of incorrect tensor index in Eq. (13.20)

Initially the synthesis used

```python
T(i,K,Q,...)
```

inside

```python
eps += Phi * T * rho
```

However Eq. (13.20) clearly contains

\[
T^{K'}_Q,
\]

not

\[
T^K_Q.
\]

The synthesis was therefore corrected to

```python
T(i,Kp,Q,...)
```

This was an important conceptual bug.

---

# 10. Array/scalar bug after switching to K'

Changing to

```python
T(...,Kp,...)
```

revealed a hidden issue:

`epsV` sometimes became a length-one NumPy array while

`epsI`,
`epsQ`,
`epsU`

remained scalars.

This produced

```text
ValueError:
setting an array element with a sequence
```

The issue was traced to mixing scalar and array objects returned by

```python
Phi_generalized(...)
```

and was fixed by consistently extracting scalars using

```python
[0]
```

or `.item()`.

---

# 11. Persistent problem: Stokes V remained Gaussian

After all fixes,

the code executed correctly,

but

Stokes V still appeared as

a Gaussian,

not the antisymmetric profile shown in Fig. 13.6.

This became the central unresolved problem.

---

# 12. Discussion of selection rules

A contradiction appeared.

Previously we had argued

that for

\[
J_l=0,\qquad
J_u=1,
\]

only

\[
K=0,2
\]

exist for the upper-level density matrix.

Yet Fig. 13.6 is explicitly computed from Eq. (13.20),

which contains K=1 contributions to Stokes V.

The resolution was

- **ρ¹ does not exist** for the upper level in this transition.
- **K'=1 does exist** inside the profile tensor.

Therefore

Stokes V originates from

\[
(K,K')
=
(2,1),
\]

not from

\[
K=1
\]

atomic polarization.

This removed the apparent contradiction.

---

# 13. Current synthesis structure

The synthesis presently loops over

```python
for K in [0,1,2]:
    for Kp in [0,1,2]:
        for Q in [-2,-1,0,1,2]:
```

using

```python
Phi_generalized(K,Kp,Q)
```

and

```python
T(i,Kp,Q,...)
```

while contracting with the Hanle-modified

```python
rho
```

obtained from

```python
apply_hanle().
```

---

# 14. Present unresolved issue

Despite

- correct tensor index (K'),
- Hanle solution,
- generalized profiles,
- K/K' summations,

the code still produces

- Gaussian I
- Gaussian Q
- Gaussian U
- Gaussian V

instead of

the expected

antisymmetric Stokes V.

---

# 15. Current leading suspicion

The most likely remaining problem is **inside the implementation of Eq. (10.40)**.

In particular,

the profile currently used is

```python
profile = 0.5 * (
    phi_transition(...)
    +
    phi_transition(...)
)
```

where

```python
phi_transition()
```

uses only the real Doppler profile

```python
phi_doppler().
```

However Eq. (10.40) in LL04 is written in terms of the complex profile

\[
\Phi(\nu)
=
\phi(\nu)
+
i\,\psi(\nu),
\]

where

- φ is the absorption profile,
- ψ is the dispersion profile.

The current implementation neglects the dispersion component entirely.

Without ψ,

the cancellation producing the characteristic antisymmetric Stokes-V profile cannot occur.

This is now considered the most probable explanation for the remaining discrepancy.
