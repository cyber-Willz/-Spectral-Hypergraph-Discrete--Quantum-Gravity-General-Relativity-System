//! Step 5 — Regge equations of motion, via the Schläfli differential identity.
//!
//! `regge.rs` builds the action S = Σ_hinges L_hinge·δ_hinge and can evaluate
//! it. It does not yet establish the fact that actually makes this an
//! honest discretization of general relativity rather than just "a formula
//! that looks like ∫R√g": that varying S with respect to a single edge
//! length L_e, holding every other edge fixed, gives
//!
//! ```text
//! dS/dL_e = delta_e            (Regge 1961; see Hartle 1985, Barrett 1994
//!                                for modern derivations)
//! ```
//!
//! i.e. the direct dependence of the OTHER hinges' deficit angles on L_e
//! (every tetrahedron touching e also touches other hinges, whose dihedral
//! angles shift when L_e moves) cancels *identically*, leaving only the
//! bare term. That cancellation is the discrete **Schläfli differential
//! identity**: for a single tetrahedron, Σ_edges L_edge·(∂θ_edge/∂L_e) = 0
//! for any edge-length variation, a purely geometric fact about tetrahedra
//! having nothing to do with gravity, which Regge noticed forces
//!
//! ```text
//! dS = sum_hinges  delta_hinge * dL_hinge
//! ```
//!
//! This is the reason the vacuum Regge field equations are simply
//! **δ_e = 0 at every interior hinge** — the exact discrete analogue of
//! R_{μν} = 0 falling out of δ(∫R√g)/δg^{μν} = 0 in the continuum. Without
//! this identity, "the Regge action" would be an aesthetically-motivated
//! functional with no particular claim to encoding a discrete field
//! equation; with it, extremizing S over edge lengths is a well-posed
//! discrete-gravity variational problem.
//!
//! This module does not re-derive Schläfli's identity symbolically (that is
//! a fixed piece of 3D Euclidean solid geometry, not something specific to
//! any hypergraph here). What it *does* do, in the spirit of the rest of
//! this crate, is verify it numerically on genuine (possibly curved)
//! simplicial complexes already built by `regge.rs`: compute ∂S/∂L_e two
//! independent ways — (a) a central finite difference of the *whole* action
//! (which implicitly sums every indirect contribution through every
//! neighboring hinge) and (b) the bare deficit angle δ_e — and confirm they
//! agree to finite-difference precision. Agreement on a *curved* (nonzero
//! deficit) configuration is the non-trivial check; on an exactly flat
//! configuration both sides are trivially ~0 and the test is much weaker.
//!
//! What this module does NOT claim:
//!   - No continuum limit statement: this is a fact about the discrete
//!     action on a fixed simplicial complex, not a claim that extremizing
//!     it converges to a solution of the continuum Einstein equations as
//!     the triangulation is refined (that convergence is itself a genuine
//!     open research question in the numerical-relativity/Regge-calculus
//!     literature, e.g. Brewin's convergence studies).
//!   - No solver: this module verifies the *identity* the equations of
//!     motion rest on, and reports per-hinge (deficit, gradient) pairs; it
//!     does not implement an extremization / relaxation routine to find
//!     vacuum (all-δ_e = 0) solutions. That is future evidentiary
//!     machinery, not claimed here.
//!   - The cosmological-constant term (`lambda != 0` in `regge_action`) is
//!     deliberately excluded from the identity checked here: Schläfli's
//!     identity is a statement about the curvature term alone. Λ ≠ 0 adds
//!     an extra `2Λ·∂Vol/∂L_e` term to the true equations of motion which
//!     this module does not separately verify.
//!
//! A real scope bug, found and fixed while writing this module (documented
//! here rather than silently worked around, matching this crate's existing
//! discipline — see RUN_LOG.txt): `regge_action` (deliberately, per its own
//! docs) sums the curvature term over *interior* hinges only (multiplicity
//! ≥ 3), because it does not implement a boundary term (the discrete
//! Gibbons–Hawking–York analogue). The full cancellation argument above
//! needs *every* edge of every tetrahedron touching e to itself be an
//! interior hinge included in that same sum — otherwise the "other" terms
//! in Σ_hinges L_h·∂δ_h/∂L_e that Schläfli's per-tetrahedron identity
//! would cancel are simply missing from S in the first place, and
//! ∂S/∂L_e ≠ δ_e. Concretely: on the flat cube (the complex used
//! throughout `regge_tests.rs`), only the single main-diagonal hinge is
//! interior — every other edge of every tetrahedron is a boundary edge —
//! so the identity numerically fails there (confirmed: it does, badly).
//! It holds only on a genuinely **closed** (boundary-free) complex, where
//! every edge of every tetrahedron is itself counted in the action sum.
//! The tests below use the boundary of a 4-simplex (5 tetrahedra, the
//! minimal closed triangulated 3-manifold — topologically S³) for exactly
//! this reason, and this restriction is the honest scope of what
//! `verify_schlafli_identity` can check against this crate's current
//! (boundary-term-free) `regge_action`.

use crate::regge::{all_tetrahedra_valid, deficit_angle, regge_action, Edge, EdgeLengths, SimplicialComplex};

/// Per-hinge comparison of the two independent ways of computing ∂S/∂L_e.
#[derive(Debug, Clone, Copy)]
pub struct EomCheck {
    pub edge: Edge,
    /// δ_e, the bare deficit angle at this hinge (the claimed value of ∂S/∂L_e).
    pub deficit_angle: f64,
    /// Central finite-difference estimate of ∂S/∂L_e using the *full*
    /// action (all hinges, all indirect dependence included).
    pub numerical_ds_dl: f64,
    pub abs_error: f64,
}

/// Return a copy of `lengths` with edge `e` shifted by `delta`, every other
/// edge held fixed exactly — the one-degree-of-freedom variation the
/// Schläfli identity is a statement about.
fn perturbed(lengths: &EdgeLengths, e: Edge, delta: f64) -> EdgeLengths {
    let mut new_lengths = lengths.clone();
    let l = new_lengths
        .lengths
        .get_mut(&e)
        .expect("edge not present in EdgeLengths");
    *l += delta;
    new_lengths
}

/// Central finite-difference estimate of ∂S_curvature/∂L_e (Λ=0), holding
/// every other edge length fixed. Returns `None` if perturbing L_e by ±h
/// would push any incident tetrahedron out of the geometrically valid
/// (Cayley-Menger positive) region — the step size `h` should be chosen
/// small enough, relative to the geometry, that this doesn't happen for an
/// interior hinge.
pub fn numerical_ds_dl(
    complex: &SimplicialComplex,
    lengths: &EdgeLengths,
    e: Edge,
    h: f64,
) -> Option<f64> {
    let plus = perturbed(lengths, e, h);
    let minus = perturbed(lengths, e, -h);
    if !all_tetrahedra_valid(complex, &plus) || !all_tetrahedra_valid(complex, &minus) {
        return None;
    }
    let s_plus = regge_action(complex, &plus, 0.0).curvature_term;
    let s_minus = regge_action(complex, &minus, 0.0).curvature_term;
    Some((s_plus - s_minus) / (2.0 * h))
}

/// Verify the Schläfli identity — ∂S/∂L_e = δ_e — at every well-posed
/// interior hinge (multiplicity ≥ 3) of `complex`, under edge lengths
/// `lengths`. `h` is the finite-difference step; a few different scales
/// should be tried and shown to converge (Richardson-style) for a genuine
/// confirmation rather than a single lucky step size.
pub fn verify_schlafli_identity(
    complex: &SimplicialComplex,
    lengths: &EdgeLengths,
    h: f64,
) -> Vec<EomCheck> {
    let mut out = Vec::new();
    for &e in &complex.edges {
        if complex.hinge_multiplicity(&e) < 3 {
            continue; // boundary edge: no interior deficit angle, out of scope
        }
        let Some(delta) = deficit_angle(complex, lengths, &e) else {
            continue;
        };
        let Some(num) = numerical_ds_dl(complex, lengths, e, h) else {
            continue; // step size too large for this hinge's local validity region
        };
        out.push(EomCheck {
            edge: e,
            deficit_angle: delta,
            numerical_ds_dl: num,
            abs_error: (num - delta).abs(),
        });
    }
    out
}

/// A vacuum (source-free) hinge under the discrete Regge field equations is
/// one with δ_e = 0 (flat). This is a pure reporting helper — it does not
/// search for or construct vacuum solutions, only classifies hinges of an
/// already-given configuration.
pub fn vacuum_hinges(checks: &[EomCheck], tol: f64) -> Vec<Edge> {
    checks
        .iter()
        .filter(|c| c.deficit_angle.abs() < tol)
        .map(|c| c.edge)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::regge::{EdgeLengths, SimplicialComplex};
    use std::collections::HashMap;

    /// The boundary of a 4-simplex: 5 vertices, 5 tetrahedral facets (each
    /// omitting one vertex), 10 edges. Combinatorially every edge {a,b}
    /// belongs to exactly the 3 facets that omit neither a nor b — i.e.
    /// EVERY edge is an interior hinge (multiplicity 3) and the complex has
    /// no boundary at all. This is the minimal closed triangulated
    /// 3-manifold (topologically S³), which is exactly the regime the
    /// Schläfli-identity cancellation needs: every edge of every
    /// tetrahedron touching the varied edge is itself included in the
    /// action sum (see the module-level doc comment on the boundary-term
    /// scope restriction this test exists to respect).
    fn boundary_of_4_simplex_tets() -> Vec<[usize; 4]> {
        vec![
            [1, 2, 3, 4],
            [0, 2, 3, 4],
            [0, 1, 3, 4],
            [0, 1, 2, 4],
            [0, 1, 2, 3],
        ]
    }

    fn dist4(a: [f64; 4], b: [f64; 4]) -> f64 {
        (0..4).map(|i| (a[i] - b[i]).powi(2)).sum::<f64>().sqrt()
    }

    fn lengths_from_coords4(complex: &SimplicialComplex, coords: &[[f64; 4]]) -> EdgeLengths {
        let mut lengths = HashMap::new();
        for &e in &complex.edges {
            lengths.insert(e, dist4(coords[e.0], coords[e.1]));
        }
        EdgeLengths { lengths }
    }

    /// 5 points in R^4 in general (non-regular, non-symmetric) position.
    /// Any 4 of them automatically satisfy the tetrahedron inequality: 4
    /// points always affinely span a subspace of dimension ≤ 3, so their
    /// pairwise distances always embed in R^3 — geometric validity of
    /// every facet is therefore guaranteed by construction, with no
    /// symmetry to accidentally mask a bug (unlike a regular 4-simplex,
    /// where every hinge would have identical deficit angle).
    fn irregular_4_simplex_coords() -> Vec<[f64; 4]> {
        vec![
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.3, 1.0, 0.0, 0.0],
            [0.2, 0.4, 1.0, 0.0],
            [0.5, 0.3, 0.2, 1.0],
        ]
    }

    /// Every one of the 10 edges of this closed complex is an interior
    /// hinge, and (by construction, no boundary) the Schläfli cancellation
    /// applies to the full curvature-term sum: ∂S/∂L_e must equal the bare
    /// deficit angle δ_e at every single edge, not just one. This is the
    /// substantive check — an irregular (no special symmetry), genuinely
    /// curved (S³-like, positive deficit angles) closed configuration.
    #[test]
    fn schlafli_identity_holds_on_closed_irregular_complex() {
        let coords = irregular_4_simplex_coords();
        let complex = SimplicialComplex::from_tetrahedra(boundary_of_4_simplex_tets());
        let lengths = lengths_from_coords4(&complex, &coords);
        assert!(all_tetrahedra_valid(&complex, &lengths));

        let checks = verify_schlafli_identity(&complex, &lengths, 1e-5);
        assert_eq!(
            checks.len(),
            complex.edges.len(),
            "every edge of a closed complex must be an interior hinge"
        );
        let mut any_curved = false;
        for c in &checks {
            if c.deficit_angle.abs() > 1e-3 {
                any_curved = true;
            }
            assert!(
                c.abs_error < 1e-4,
                "edge {:?}: deficit={}, numerical dS/dL={}, err={}",
                c.edge,
                c.deficit_angle,
                c.numerical_ds_dl,
                c.abs_error
            );
        }
        assert!(
            any_curved,
            "expected a genuinely curved (nonzero-deficit) configuration"
        );
    }

    /// The regular 4-simplex (all 10 edges equal length) gives, by
    /// symmetry, the same nonzero deficit angle at every hinge — a
    /// simpler, fully symmetric curved case, cross-checked against the
    /// irregular complex above with an independent construction.
    #[test]
    fn schlafli_identity_holds_on_regular_4_simplex_boundary() {
        // A regular 4-simplex has all 10 pairwise distances equal; rather
        // than embedding one explicitly, just assign every edge the same
        // length directly -- geometric validity (Cayley-Menger positivity)
        // for the equal-edge-length regular tetrahedron case is already
        // established by `regge_tests.rs`'s
        // `regular_tetrahedron_dihedral_angle_matches_closed_form`.
        let complex = SimplicialComplex::from_tetrahedra(boundary_of_4_simplex_tets());
        let mut lengths = HashMap::new();
        for &e in &complex.edges {
            lengths.insert(e, 1.0);
        }
        let lengths = EdgeLengths { lengths };
        assert!(all_tetrahedra_valid(&complex, &lengths));

        let checks = verify_schlafli_identity(&complex, &lengths, 1e-5);
        assert_eq!(checks.len(), 10);
        let first_deficit = checks[0].deficit_angle;
        assert!(first_deficit.abs() > 1e-3, "expected a curved configuration");
        for c in &checks {
            // Full vertex-transitive symmetry: every hinge has the same deficit.
            assert!((c.deficit_angle - first_deficit).abs() < 1e-9);
            assert!(
                c.abs_error < 1e-4,
                "edge {:?}: deficit={}, numerical dS/dL={}, err={}",
                c.edge,
                c.deficit_angle,
                c.numerical_ds_dl,
                c.abs_error
            );
        }
    }

    /// `vacuum_hinges` should correctly report zero vacuum (flat) hinges
    /// on this genuinely, uniformly curved closed complex.
    #[test]
    fn vacuum_hinges_reports_none_on_curved_closed_complex() {
        let coords = irregular_4_simplex_coords();
        let complex = SimplicialComplex::from_tetrahedra(boundary_of_4_simplex_tets());
        let lengths = lengths_from_coords4(&complex, &coords);
        let checks = verify_schlafli_identity(&complex, &lengths, 1e-5);
        assert_eq!(vacuum_hinges(&checks, 1e-3).len(), 0);
    }
}
