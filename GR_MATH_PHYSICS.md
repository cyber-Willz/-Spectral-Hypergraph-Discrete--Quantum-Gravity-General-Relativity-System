# Mathematics and Physics of the General Relativity System

This document lays out, in one place, the actual derivations behind every
piece of GR machinery in this crate — the continuum tensor-calculus engine
(`tensor_calculus.rs`), the exact solutions (`metrics.rs`), the geodesic
integrator (`geodesics.rs`), and the discrete Regge-calculus side
(`regge.rs`, `regge_eom.rs`). It is written to be checked, not taken on
faith: every claim below is either (a) a standard textbook proof, reproduced
here in enough detail to verify by hand, or (b) an explicit statement that
something is a *numerical* check, not a proof, together with the exact test
that performs the check and the precision it achieved.

A word on the title. "Proof of the system" is not a single meaningful
target — a piece of software is not a theorem. What follows is: the
mathematical proofs that the *definitions* used are self-consistent
(e.g. that the geodesic equation follows from an extremal-length
variational principle), and the numerical evidence that the *code*
correctly implements those definitions (e.g. that the computed Kretschmann
scalar matches the closed-form answer to four significant figures). Anywhere
this document says "proof," it means an actual derivation. Anywhere it says
"verified," it means a specific test in the codebase, cited by name, with
its measured error.

---

## 1. Conventions

Fixed throughout `tensor_calculus.rs`, `metrics.rs`, `geodesics.rs`:

- Signature `(-,+,+,+)`.
- Units `c = G = 1` (so a Schwarzschild mass parameter is directly the
  Schwarzschild radius `r_s = 2M`).
- Coordinates `x = [x^0, x^1, x^2, x^3]`, indices `a,b,c,d,e,... ∈ {0,1,2,3}`,
  Einstein summation convention.
- Riemann tensor sign convention (matching Wald, *General Relativity*,
  eq. 3.2.3):

```
R^a_{bcd} = ∂_c Γ^a_{db} − ∂_d Γ^a_{cb} + Γ^a_{ce} Γ^e_{db} − Γ^a_{de} Γ^e_{cb}
R_{bd}    = R^a_{bad}                      (Ricci: contract 1st & 3rd indices)
```

Every formula below uses these conventions consistently; a different but
equally standard convention (e.g. MTW's) would flip some overall signs, so
this section is load-bearing for everything that follows.

---

## 2. The Levi-Civita connection and Christoffel symbols

**Claim.** For a metric `g_{ab}`, there is a unique torsion-free connection
`∇` compatible with the metric (`∇_c g_{ab} = 0`), and its coefficients in a
coordinate basis are

```
Γ^a_{bc} = (1/2) g^{ad} ( ∂_b g_{dc} + ∂_c g_{db} − ∂_d g_{bc} ).
```

**Proof (existence and uniqueness — the "fundamental theorem of Riemannian
geometry").** Write out metric compatibility for the three index
permutations of `(a,b,c)`:

```
∂_a g_{bc} = Γ^d_{ab} g_{dc} + Γ^d_{ac} g_{bd}          (i)
∂_b g_{ca} = Γ^d_{bc} g_{da} + Γ^d_{ba} g_{cd}          (ii)
∂_c g_{ab} = Γ^d_{ca} g_{db} + Γ^d_{cb} g_{ad}          (iii)
```

Compute `(i) + (ii) − (iii)`. Using symmetry of `Γ^d_{bc}` in its lower
indices (torsion-free: `Γ^d_{bc} = Γ^d_{cb}`, which holds because torsion is
defined as `T^d_{bc} = Γ^d_{bc} − Γ^d_{cb}` and is assumed zero) and
symmetry of `g_{ab}`, every term except one pair cancels, leaving

```
∂_a g_{bc} + ∂_b g_{ca} − ∂_c g_{ab} = 2 Γ^d_{ab} g_{dc}.
```

Contracting both sides with `g^{ec}` and renaming indices gives exactly the
formula above. This also proves *uniqueness*: the derivation only used
compatibility and vanishing torsion, so any connection satisfying both must
have these coefficients. ∎

This is implemented verbatim in `tensor_calculus::christoffel`, except that
`∂g` is not computed symbolically — it's a central finite difference,
`(g(x + h·ê_c) − g(x − h·ê_c)) / 2h`. This substitution is the single source
of numerical (as opposed to mathematical) error in the entire engine; every
downstream quantity (Riemann, Ricci, Einstein, Kretschmann) inherits it.

**Verified:** `tensor_calculus::tests::constant_metric_has_zero_christoffel`
checks the trivial case (constant metric ⟹ exact zero, no truncation error
since the derivative of a constant is exactly zero regardless of `h`).

---

## 3. The Riemann curvature tensor

**Definition (coordinate-free).** For vector fields `X, Y, Z`,

```
R(X,Y)Z = ∇_X ∇_Y Z − ∇_Y ∇_X Z − ∇_{[X,Y]} Z.
```

This measures the failure of second covariant derivatives to commute — the
precise sense in which "curvature is the obstruction to flatness."

**Claim.** In a coordinate basis (where `[∂_b, ∂_c] = 0`), this reduces to

```
R^a_{bcd} = ∂_c Γ^a_{db} − ∂_d Γ^a_{cb} + Γ^a_{ce} Γ^e_{db} − Γ^a_{de} Γ^e_{cb}.
```

**Proof.** Apply the definition to coordinate basis vectors
`X=∂_c, Y=∂_d, Z=∂_b` (so `[X,Y]=0`, dropping the last term):

```
∇_c ∇_d ∂_b = ∇_c (Γ^e_{db} ∂_e) = (∂_c Γ^e_{db}) ∂_e + Γ^e_{db} ∇_c ∂_e
            = (∂_c Γ^e_{db}) ∂_e + Γ^e_{db} Γ^a_{ce} ∂_a.
```

Relabel the dummy index in the first term (`e → a`) and swap `c ↔ d` for the
second application, then subtract:

```
R(∂_c,∂_d)∂_b = [∂_c Γ^a_{db} − ∂_d Γ^a_{cb} + Γ^a_{ce}Γ^e_{db} − Γ^a_{de}Γ^e_{cb}] ∂_a,
```

and reading off the coefficient of `∂_a` (which is `R^a_{bcd}` by
definition of the component convention `R(∂_c,∂_d)∂_b = R^a_{bcd}∂_a`) gives
the claimed formula. ∎

`tensor_calculus::riemann` implements this directly: it builds `Γ` at `x`,
then numerically differentiates `Γ` itself (a *second* finite difference of
the metric — the dominant error source, since round-off in the inner
difference gets amplified by the outer one). The module's doc comment
states the expected precision (`~1e-4`–`1e-6` relative, for the recommended
`h = 1e-4`) as an empirical claim, not a derived error bound — deriving a
tight bound would require tracking round-off through the nested difference,
which this crate doesn't attempt; instead it's checked directly against
closed-form answers (§5–6).

**Verified:** `tensor_calculus::tests::minkowski_is_flat` — every component
of `R^a_{bcd}` for the (exactly flat) Minkowski metric comes out
`< 1e-6` in absolute value, which is the finite-difference noise floor for
`h = 1e-4`, not a real curvature signal.

---

## 4. Ricci tensor, Ricci scalar, Einstein tensor, Kretschmann scalar

Standard contractions, all implemented directly from their definitions:

```
R_{bd}  = R^a_{bad}                     (Ricci tensor)
R       = g^{bd} R_{bd}                 (Ricci scalar)
G_{ab}  = R_{ab} − (1/2) R g_{ab}       (Einstein tensor)
K       = R_{abcd} R^{abcd}             (Kretschmann scalar)
```

**Why the Kretschmann scalar matters as a check.** The Ricci tensor is only
part of the Riemann tensor's information — in vacuum (`R_{ab}=0`) the Ricci
tensor and scalar are identically zero, but spacetime can still be curved
(tidal forces, e.g. outside a star). The Kretschmann scalar contracts the
*full* Riemann tensor and is nonzero in vacuum whenever there is genuine
curvature; it is the standard invariant used to diagnose a true curvature
singularity (it diverges at `r=0` in Schwarzschild, unlike the coordinate
singularity at `r=r_s`, which is only a bad choice of coordinates — the
distinction is a classical result: `K → ∞` as `r→0` but `K` stays finite at
`r=r_s`). This is why `metrics.rs` cross-checks Kretschmann and not just
Ricci: agreement on Kretschmann is a genuine test of the Riemann tensor
computation, not just "everything numerically vanishes."

**Bianchi identity (stated, not separately verified numerically here).**
The second Bianchi identity `∇_{[e}R_{ab]cd}=0` implies, after contraction,
the contracted Bianchi identity `∇^a G_{ab} = 0`. This is *the* reason the
Einstein tensor (rather than the Ricci tensor) appears in the field
equations `G_{ab}=8πT_{ab}`: it guarantees `∇^a T_{ab}=0` automatically,
consistent with local conservation of energy-momentum. This crate computes
`G_{ab}` (§ above) but does not separately verify `∇^a G_{ab}=0` as a test —
that would require yet another numerical derivative layer on top of an
already twice-differentiated quantity, compounding truncation error faster
than it would usefully constrain anything not already checked by the
Ricci/Kretschmann tests above. Flagging this explicitly rather than
implying it was checked.

---

## 5. Schwarzschild: derivation sketch, and what's actually verified

**Setup.** Assume a static, spherically symmetric vacuum metric. The most
general such metric can be written (a standard reduction, via the existence
of a timelike Killing vector and `SO(3)` symmetry orbits) as

```
ds² = −f(r) dt² + dr²/h(r) + r²(dθ² + sin²θ dφ²)
```

for functions `f, h` to be determined by `R_{ab}=0`.

**Result (Birkhoff's theorem + explicit solution).** Substituting this
ansatz into `R_{ab}=0` and solving the resulting ODEs (a standard but
lengthy computation — see e.g. Wald §6.1 or MTW §31 — not reproduced
component-by-component here since it is a long but entirely mechanical
calculation, not an insightful one) gives `f(r)=h(r)=1−r_s/r` for a constant
`r_s` identified (by matching the Newtonian limit `g_{tt}≈−(1+2Φ)`,
`Φ=−M/r`) with `r_s=2M`:

```
ds² = −(1−r_s/r) dt² + dr²/(1−r_s/r) + r²dθ² + r²sin²θ dφ².
```

Birkhoff's theorem additionally states this is the *unique* spherically
symmetric vacuum solution (even without assuming staticity in advance) —
the vacuum field equations alone rule out a monopole gravitational wave.
Neither the ODE solution nor Birkhoff's theorem is re-derived or
re-proven in this codebase; they are taken from the literature and used as
the closed-form target that the numerical engine is checked against.

**What is actually computed and checked here:**

1. `metrics::schwarzschild` implements the metric above as a callback.
2. `tensor_calculus::curvature_at` computes `R_{ab}` numerically from it.
3. `metrics::tests::schwarzschild_is_ricci_flat` checks `|R_{ab}| < 1e-3`
   at `r = 10 r_s` — i.e., that the *specific metric formula in the code*
   actually solves the vacuum equations, which is not automatic (a typo in
   the metric, e.g. a wrong power of `r`, would not be caught by any type
   system and would silently produce a non-solution).
4. The exact Kretschmann scalar for this solution is a known closed form,

   ```
   K = 48 M² / r⁶ = 12 r_s² / r⁶     (since M = r_s/2)
   ```

   (standard result, e.g. MTW). `metrics::schwarzschild_kretschmann_exact`
   encodes this, and `metrics::tests::schwarzschild_kretschmann_matches_closed_form`
   checks the numerically-computed `K` against it at `r = 5, 10, 25` (in
   units of `r_s`), passing at relative error `< 1e-2` — measured, in the
   live run recorded in `session_output/gr_demo_output.txt`, at **≤0.008%**
   across `r = 3` to `50 r_s`.

---

## 6. FRW: derivation sketch, and what's actually verified

**Setup (cosmological principle).** Assume spatial homogeneity and isotropy
at each fixed time. This forces the spatial metric at fixed `t` to be a
maximally symmetric 3-space of constant curvature `k ∈ {−1,0,+1}` (up to
rescaling), and the full metric to take the Robertson–Walker form

```
ds² = −dt² + a(t)² [ dχ²/(1−kχ²) + χ²(dθ²+sin²θ dφ²) ],
```

with `a(t)` (the scale factor) left undetermined by symmetry alone — it is
fixed by the Einstein equations *given* a matter content, via the Friedmann
equations. `metrics::frw` implements exactly this metric, with `a(t)` as an
arbitrary caller-supplied function — deliberately not tied to any specific
matter sector (see §"what is not claimed" in the module docs).

**Claim (Ricci scalar of the Robertson-Walker metric).**

```
R = 6 [ ä/a + (ȧ/a)² + k/a² ].
```

**Proof sketch.** This follows from a direct (if long) computation of
`Γ^a_{bc}` for the metric above, then `R^a_{bcd}`, then contracting — the
same three-step chain as §2–4, just carried out symbolically for this
specific metric rather than a generic one. The nonzero Christoffel symbols
are (with `χ` written `r` for brevity, `H ≡ ȧ/a`):

```
Γ^t_{rr} = a ȧ/(1−kr²),   Γ^t_{θθ} = a ȧ r²,   Γ^t_{φφ} = a ȧ r² sin²θ
Γ^r_{tr} = Γ^θ_{tθ} = Γ^φ_{tφ} = H
Γ^r_{rr} = kr/(1−kr²),    Γ^r_{θθ} = −r(1−kr²),  Γ^r_{φφ} = −r(1−kr²) sin²θ
Γ^θ_{rθ} = Γ^φ_{rφ} = 1/r,   Γ^θ_{φφ} = −sinθ cosθ,   Γ^φ_{θφ} = cotθ.
```

Contracting these through the Riemann/Ricci formulas of §3–4 (mechanical
but lengthy; carried out here symbolically off-line, not by the crate's
finite-difference engine, precisely so it serves as an independent target)
gives the diagonal Ricci tensor components

```
R_{tt} = −3 ä/a
R_{rr} = [a ä + 2ȧ² + 2k] / (1−kr²)
R_{θθ} = r² [aä + 2ȧ² + 2k]
R_{φφ} = r² sin²θ [aä + 2ȧ² + 2k],
```

and contracting with `g^{ab}` (`g^{tt}=−1`, `g^{rr}=(1−kr²)/a²`,
`g^{θθ}=1/(a²r²)`, `g^{φφ}=1/(a²r²sin²θ)`):

```
R = g^{tt}R_{tt} + g^{rr}R_{rr} + g^{θθ}R_{θθ} + g^{φφ}R_{φφ}
  = 3ä/a + 3(aä+2ȧ²+2k)/a²
  = 6[ ä/a + (ȧ/a)² + k/a² ].  ∎
```

**What is actually computed and checked here.** For the matter-dominated
flat (`k=0`) exact Friedmann solution `a(t) = t^{2/3}` (chosen because
`ȧ = (2/3)t^{-1/3}` and `ä = -(2/9)t^{-4/3}` are simple closed forms):

- `metrics::tests::frw_ricci_scalar_matches_closed_form_matter_dominated`
  checks the numerically-computed `R` against the formula above at `t=2.5`,
  passing at `<1e-2` relative error — measured in the live run at
  **≤0.001%** across `t = 1, 2.5, 5, 10`.
- `metrics::tests::frw_ricci_scalar_is_spatially_homogeneous` independently
  checks that `R` computed at three unrelated `(χ,θ,φ)` points at fixed `t`
  agree with each other — a structural consequence of the cosmological
  principle that the closed-form check above would *not* by itself catch
  breaking (e.g. a metric with a coordinate-dependent bug that happened to
  give the right value at one specific test point).

---

## 7. The geodesic equation, from a variational principle

**Claim.** A curve `x^a(λ)` extremizing the proper-length (or, for null
curves, an appropriately regularized) functional

```
S[x] = ∫ L dλ,   L = g_{ab}(x) ẋ^a ẋ^b   (ẋ ≡ dx/dλ)
```

satisfies the geodesic equation

```
ẍ^a + Γ^a_{bc} ẋ^b ẋ^c = 0.
```

**Proof.** Euler–Lagrange for `L`:

```
d/dλ (∂L/∂ẋ^e) − ∂L/∂x^e = 0.
```

`∂L/∂ẋ^e = 2 g_{ea} ẋ^a`, so `d/dλ(...) = 2 g_{ea} ẍ^a + 2 (∂_c g_{ea}) ẋ^c ẋ^a`.
`∂L/∂x^e = (∂_e g_{ab}) ẋ^a ẋ^b`. Substituting and dividing by 2:

```
g_{ea} ẍ^a + (∂_c g_{ea}) ẋ^c ẋ^a − (1/2)(∂_e g_{ab}) ẋ^a ẋ^b = 0.
```

Symmetrizing the middle term over `(a,c)` (valid since it's contracted with
the symmetric `ẋ^aẋ^c`) and combining with the last term reproduces exactly
the bracket in the Christoffel formula of §2:

```
g_{ea} ẍ^a + (1/2)(∂_c g_{ea} + ∂_a g_{ec} − ∂_e g_{ac}) ẋ^a ẋ^c = 0.
```

Raising the free index with `g^{fe}` gives `ẍ^f + Γ^f_{ac} ẋ^a ẋ^c = 0`. ∎

This is implemented directly in `geodesics::rk4_step` (via `rhs`, which
computes `Γ^a_{bc}` numerically at each RK4 stage using the *same*
`tensor_calculus::christoffel` function as the curvature engine — the
geodesic integrator and the curvature computations are not independent
code paths, so a bug in `christoffel` would show up in both, which is why
the cross-checks in §5–6 also indirectly validate the geodesic integrator's
force term).

**Verified (straight-line case, exact):**
`geodesics::tests::minkowski_geodesic_is_a_straight_line` — in flat space
`Γ≡0`, so the geodesic equation reduces to `ẍ=0`; the RK4-integrated
trajectory over 100 steps agrees with the exact linear solution to
`< 1e-6` (pure RK4 discretization error on a case with zero curvature
"signal" to get wrong).

---

## 8. Conserved quantities from Killing vectors

**Claim.** If `ξ^a` is a Killing vector (`∇_{(a}ξ_{b)}=0`, i.e. the metric
is invariant along its flow), then `Q ≡ ξ_a ẋ^a` is constant along any
geodesic.

**Proof.**

```
dQ/dλ = ẋ^b ∇_b (ξ_a ẋ^a) = ẋ^b ẋ^a ∇_b ξ_a + ξ_a ẋ^b ∇_b ẋ^a.
```

The second term vanishes because `ẋ^b∇_bẋ^a=0` is exactly the geodesic
equation. The first term is `ẋ^aẋ^b ∇_b ξ_a`, a contraction of the
symmetric tensor `ẋ^aẋ^b` with `∇_bξ_a`; since `∇_{(b}ξ_{a)}=0` means
`∇_bξ_a` is *antisymmetric*, the contraction of a symmetric tensor with an
antisymmetric one vanishes identically. Hence `dQ/dλ=0`. ∎

**Application to Schwarzschild.** `∂_t` and `∂_φ` are Killing vectors of
the Schwarzschild metric (it doesn't depend on `t` or `φ`), giving the two
conserved quantities used throughout `geodesics.rs`:

```
E ≡ −g_{tt} ẋ^t = (1−r_s/r) ẋ^t      (energy per unit mass)
L ≡ g_{φφ} ẋ^φ = r² sin²θ ẋ^φ          (angular momentum per unit mass)
```

**Why this is a meaningful test rather than a tautology.** Nothing in
`geodesics::rk4_step` computes or enforces `E` or `L` — it only ever
evaluates the local geodesic equation via numerically-differentiated
Christoffel symbols. Their near-constancy along an integrated trajectory is
therefore an *emergent* check: it can only hold if (a) the Christoffel
symbols are numerically correct, and (b) the integration is accurate enough
not to drift off the true geodesic. Either kind of bug (a formula error or
an integration-accuracy problem) would show up as `E` or `L` visibly
drifting.

**Verified:**
`geodesics::tests::schwarzschild_geodesic_conserves_energy_and_angular_momentum`
— for an eccentric (non-circular) timelike geodesic integrated over 3000
RK4 steps, both `E` and `L` stay within `1%` of their initial values at
every sampled point; the live run records them holding to 8 decimal places
(`E=1.05000000`, `L=0.60000000`) across the entire run, i.e. drift well
below the `1%` test tolerance.

Separately, `geodesics::norm` checks the (unrelated, but also structural)
fact that `g_{ab}ẋ^aẋ^b` is constant along *any* geodesic — this follows
from the same computation as above with `ξ_a` replaced by `ẋ_a` itself
(`d/dλ(g_{ab}ẋ^aẋ^b) = 2ẋ^bẋ^a∇_bẋ_a = 0` directly from the geodesic
equation, no Killing-vector assumption needed at all). Verified in
`geodesics::tests::schwarzschild_timelike_geodesic_conserves_norm`.

---

## 9. Light bending: full derivation and the bug it exposed

**Setup.** A photon (null geodesic, `g_{ab}ẋ^aẋ^b=0`) moving in the
equatorial plane (`θ=π/2`, preserved by symmetry once `ẋ^θ=0` initially) of
Schwarzschild, with conserved `E` and `L` as in §8. The null condition gives

```
−(1−r_s/r) ṫ² + ṙ²/(1−r_s/r) + r²φ̇² = 0.
```

Substituting `ṫ=E/(1−r_s/r)`, `φ̇=L/r²`, and defining the impact parameter
`b ≡ L/E`:

```
ṙ² = E² − (L²/r²)(1−r_s/r) = E² [ 1 − (b²/r²)(1−r_s/r) ].
```

**Weak-field deflection formula.** Changing variables to `u=1/r` and
expanding to first order in `r_s/b` (standard perturbative treatment — see
e.g. Hartle, *Gravity*, ch. 9, or Weinberg §8.5; not re-derived digit-by-
digit here since it is a standard textbook exercise in solving the orbit
equation `d²u/dφ² + u = 3Mu² ` perturbatively) gives the total deflection
of a photon coming from and returning to spatial infinity:

```
Δφ = 4GM/(c²b) = 2r_s/b     (in the c=G=1 units used here).
```

This is the number `schwarzschild_light_deflection` is checked against.

**What the code actually does (not the perturbative expansion above, the
full nonlinear geodesic equation):** sets up exact initial conditions at
finite radius `r0` matching a given `E=1, L=b` (solving the null condition
above for the initial `ṙ`, taking the incoming/negative root), integrates
the *full, unapproximated* geodesic equation via RK4 through closest
approach and back out to `r=r0`, and measures the total angle swept in
`φ`.

**The bug, and its fix (documented here because it is a genuine
mathematical subtlety, not a coding slip).** The naive comparison is
`Δφ_measured = φ_final − π`, reasoning that an undeflected (flat-space)
trajectory sweeps exactly `π` between its two asymptotes at infinity. But
the code integrates between two points at *finite* `r=r0`, not between true
asymptotes — and at finite `r0`, even in flat space, the swept angle is
*not* `π`. Concretely: a straight line in flat space at perpendicular
distance `b` from the origin, between the two points where it crosses
radius `r0`, sweeps

```
Δφ_flat(r0,b) = π − 2 arcsin(b/r0),
```

which is derivable directly from the triangle formed by the origin, the
point of closest approach, and either endpoint (a right triangle with
hypotenuse `r0`, opposite side `b`, so the angle at the origin between the
perpendicular and the endpoint is `arcsin(b/r0)`, and the total swept angle
is `π` minus twice that). This equals `π` only in the limit `r0→∞`. At the
test's original parameters (`b=50, r0=200`, so `b/r0=0.25`), this
correction is `2·arcsin(0.25) ≈ 0.505` radians — *larger* than the actual
GR deflection being measured (`2r_s/b=0.04` for `r_s=1`) — which is exactly
what produced a wrong-sign, order-of-magnitude-wrong result on the first
implementation attempt. The fix subtracts the correct finite-`r0` flat
baseline instead of bare `π`:

```rust
let flat_baseline = std::f64::consts::PI - 2.0 * (b / r0).asin();
state.x[3] - flat_baseline
```

**Verified:**
`geodesics::tests::light_bending_matches_weak_field_prediction` and
`geodesics::tests::light_bending_deflection_is_stable_across_r0_choices`
— agreement with `2r_s/b` at `<10%` (the test tolerance, chosen because the
weak-field formula itself is only a first-order approximation whose error
is `O((r_s/b)²) ≈ 4×10⁻⁴` for `b=50r_s`, so most of the tolerance budget is
for finite integration step size, not formula error); measured in the live
run at **2.5–2.9%** across three different `r0` choices (`200, 500, 1000`
in units of `r_s`), confirming the fix generalizes rather than being tuned
to one specific `r0`.

---

## 10. The discrete side: Regge calculus (for completeness)

The continuum machinery above (§1–9) is entirely independent of the
discrete Regge-calculus code in `regge.rs`/`regge_eom.rs`; the two are not
currently bridged (see README). For completeness, the discrete side's core
mathematical claim:

**Regge's theorem (deficit angle as discrete curvature).** On a simplicial
complex with flat simplices, all curvature is concentrated on the
codimension-2 "hinges" (edges, in 4D), measured by the *deficit angle*
`δ_h = 2π − Σ θ_h` (sum of dihedral angles at that hinge over all simplices
meeting there). The Regge action

```
S_Regge = Σ_h A_h δ_h
```

(sum over hinges of hinge volume times deficit angle) is the simplicial
discretization of the Einstein–Hilbert action `∫ R √g d⁴x`, converging to
it in a refinement limit (Regge 1961; Cheeger–Müller–Schrader 1984 for the
rigorous convergence statement). This crate does not re-derive or attempt
to re-prove Regge's original result; it takes the Regge action as given and
numerically verifies a *consequence* of it — the Schläfli identity
`dS/dL_e = δ_e` (the variation of the action with respect to an edge length
equals exactly the deficit angle at that edge, with all other terms
vanishing by a geometric identity specific to simplices) — to near machine
precision on closed complexes. See `regge_eom.rs` and the main README for
that verification's own detail and its explicitly stated limitation (open
complexes require a discrete Gibbons–Hawking–York boundary term that is not
yet implemented).

---

## 11. Summary table: proof vs. numerical verification

| Claim | Status | Where |
|---|---|---|
| Christoffel formula from metric compatibility | Proved (§2) | `tensor_calculus::christoffel` |
| Riemann tensor formula from `∇` non-commutativity | Proved (§3) | `tensor_calculus::riemann` |
| Ricci/Einstein/Kretschmann definitions | Definitional, no proof needed | `tensor_calculus.rs` |
| Minkowski is flat | Proved trivially + verified `<1e-6` | `tests::minkowski_is_flat` |
| Schwarzschild solves vacuum Einstein eqs | Proved in literature (not here); verified `R_ab<1e-3`, live run `≤0.008%` on Kretschmann | `metrics::tests::schwarzschild_*` |
| FRW Ricci scalar formula | Proved (§6) + verified, live run `≤0.001%` | `metrics::tests::frw_*` |
| Geodesic equation from variational principle | Proved (§7) | `geodesics::rk4_step` |
| Killing vectors ⟹ conserved quantities | Proved (§8) + verified (`E,L` const. to 8 d.p. over 3000 steps) | `geodesics::tests::*conserves*` |
| Weak-field light bending `Δφ=2r_s/b` | Proved in literature (not here, standard perturbative result); verified `2.5–2.9%` across 3 values of `r0` | `geodesics::tests::light_bending_*` |
| Regge action ≈ Einstein-Hilbert action | Proved in literature (Regge 1961, not here) | `regge.rs` |
| Schläfli identity on closed complexes | Verified to near machine precision | `regge_eom.rs` |
| Discrete/continuum bridge | **Not implemented** | — |
| Diffeomorphism invariance of the discrete construction | **Not proved or checked** | — |
| Full Einstein field equations with matter (either side) | **Not implemented** | — |

The bottom three rows are repeated from the README deliberately: this
document collects the actual mathematics, but it does not expand what has
been implemented, and listing the boundaries again here (rather than only
in the README) is meant to keep this file from being read as a stronger
claim than the code supports.
