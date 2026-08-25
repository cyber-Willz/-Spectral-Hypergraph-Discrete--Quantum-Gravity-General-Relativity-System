//! Cheeger constant (conductance) vs. spectral gap — Cheeger's inequality.
//!
//! `continuum_limit.rs` already reports a Ramanujan/Alon–Boppana diagnostic
//! (how close the non-backtracking spectrum sits to the theoretical
//! expander bound) as the honest proxy for "how expander-like is this
//! graph". This module adds the *combinatorial* side of that same claim:
//! the Cheeger constant (edge conductance) h(G), which measures expansion
//! directly (can you cut the graph into two large pieces using few edges?)
//! with no spectral theory involved, and cross-checks it against the
//! `laplacian::spectrum` spectral gap λ₁ via **Cheeger's inequality**
//! (discrete form, symmetric-normalized Laplacian, e.g. F. Chung,
//! *Spectral Graph Theory*, Thm 2.2):
//!
//! ```text
//! lambda_1 / 2   <=   h(G)   <=   sqrt(2 * lambda_1)
//! ```
//!
//! This is a genuine, two-sided, provable bound relating a purely
//! combinatorial quantity to a purely spectral one — exactly the kind of
//! independent cross-check ("does the spectral machinery actually track
//! what it's supposed to track?") the rest of this crate uses the Ihara/
//! Bass zeta agreement and the exact/SLQ heat-trace agreement for. A graph
//! failing this inequality would indicate a bug in either `laplacian.rs` or
//! this module, not a new physics result — it's a mathematical theorem, not
//! an experiment.
//!
//! What this module does NOT claim:
//!   - `cheeger_constant_exact` is a brute-force minimum over all 2^(N-1)-1
//!     nontrivial bipartitions. This is only tractable for small N (a few
//!     tens of vertices); it is exact evidentiary machinery for
//!     small/moderate test graphs, not a scalable expansion estimator. No
//!     approximation algorithm (e.g. spectral partitioning / sweep cuts) is
//!     implemented here.
//!   - Cheeger's inequality itself is a fixed theorem of spectral graph
//!     theory, not something this crate discovers; what's implemented is
//!     the computation of both sides on graphs already built by
//!     `hypergraph.rs`/`continuum_limit.rs`, to confirm the two pieces of
//!     machinery agree with each other and with the theorem.

use crate::hypergraph::WeightedGraph;
use crate::laplacian::spectrum;

/// Conductance of a bipartition (S, V\S): |edges crossing the cut| divided
/// by min(vol(S), vol(V\S)), where vol(A) = sum of degrees of vertices in A
/// (the standard normalized-Laplacian-compatible volume convention).
pub fn conductance(g: &WeightedGraph, in_s: &[bool]) -> f64 {
    let mut cut_weight = 0.0;
    let mut vol_s = 0.0;
    let mut vol_not_s = 0.0;
    let degrees: Vec<f64> = (0..g.n).map(|v| g.degree(v)).collect();
    for v in 0..g.n {
        if in_s[v] {
            vol_s += degrees[v];
        } else {
            vol_not_s += degrees[v];
        }
    }
    for &((u, v), w) in &g.edges {
        if in_s[u] != in_s[v] {
            cut_weight += w;
        }
    }
    let denom = vol_s.min(vol_not_s);
    if denom <= 1e-14 {
        return f64::INFINITY; // degenerate (empty side): not a real bipartition
    }
    cut_weight / denom
}

/// Exact Cheeger constant h(G) = min over nonempty proper S ⊆ V of
/// conductance(S). Brute force over all 2^(N-1) - 1 bipartitions (fixing
/// vertex 0 ∈ S to avoid double-counting S vs V\S) — exponential, so this
/// is only meant for small graphs (N up to the low twenties).
pub fn cheeger_constant_exact(g: &WeightedGraph) -> f64 {
    assert!(
        g.n >= 2 && g.n <= 24,
        "brute-force Cheeger constant is only tractable for small graphs (2..=24 vertices), got n={}",
        g.n
    );
    let n = g.n;
    let mut best = f64::INFINITY;
    // Fix vertex 0's membership to `true` (in S) by only iterating masks
    // with bit 0 set; every bipartition is then counted exactly once.
    for mask in 1u32..(1u32 << (n - 1)) {
        let mut in_s = vec![false; n];
        in_s[0] = true;
        for v in 1..n {
            if (mask >> (v - 1)) & 1 == 1 {
                in_s[v] = true;
            }
        }
        let phi = conductance(g, &in_s);
        if phi < best {
            best = phi;
        }
    }
    best
}

/// Result of cross-checking the exact Cheeger constant against the
/// normalized-Laplacian spectral gap λ₁ (the smallest nonzero eigenvalue,
/// i.e. `spectrum.eigenvalues[1]` for a connected graph).
#[derive(Debug, Clone, Copy)]
pub struct CheegerCheck {
    pub lambda_1: f64,
    pub h: f64,
    pub lower_bound: f64, // lambda_1 / 2
    pub upper_bound: f64, // sqrt(2 * lambda_1)
    pub holds: bool,
}

/// Compute both sides of Cheeger's inequality for `g` and confirm
/// `lower_bound <= h <= upper_bound` (with a small numerical tolerance for
/// the brute-force minimum and the dense eigensolver both carrying their
/// own float error).
pub fn cheeger_inequality_check(g: &WeightedGraph) -> CheegerCheck {
    let spec = spectrum(g, true);
    let lambda_1 = spec.eigenvalues.get(1).copied().unwrap_or(0.0);
    let h = cheeger_constant_exact(g);
    let lower_bound = lambda_1 / 2.0;
    let upper_bound = (2.0 * lambda_1).sqrt();
    let tol = 1e-9;
    let holds = h + tol >= lower_bound && h - tol <= upper_bound;
    CheegerCheck {
        lambda_1,
        h,
        lower_bound,
        upper_bound,
        holds,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hypergraph::Hypergraph;

    /// The N-cycle has a known exact Cheeger constant: the best cut always
    /// removes exactly 2 edges, splitting into two arcs of nearly equal
    /// volume (each vertex has degree 2, so vol(S) = 2|S|). For even N,
    /// the balanced cut gives conductance = 2 / (2 * N/2) = 2/N.
    #[test]
    fn cheeger_constant_matches_known_cycle_value() {
        let n = 10;
        let mut hg = Hypergraph::new(n);
        for i in 0..n {
            hg.add_hyperedge(vec![i, (i + 1) % n], 1.0);
        }
        let g = hg.clique_expand();
        let h = cheeger_constant_exact(&g);
        let expected = 2.0 / n as f64;
        assert!(
            (h - expected).abs() < 1e-9,
            "C_10 cycle: h={h}, expected {expected}"
        );
    }

    /// The complete graph K_n has Cheeger constant achieved by the most
    /// balanced cut: for a k/(n-k) split, cut weight = k(n-k), vol(S) =
    /// k(n-1), vol(not S) = (n-k)(n-1); minimizing over k gives the
    /// well-known value h(K_n) = ceil(n/2) / (n-1) for the balanced split
    /// (here checked numerically rather than asserted from a formula, to
    /// keep this test itself an independent computation).
    #[test]
    fn cheeger_constant_positive_and_bounded_on_complete_graph() {
        let n = 6;
        let mut hg = Hypergraph::new(n);
        for i in 0..n {
            for j in (i + 1)..n {
                hg.add_hyperedge(vec![i, j], 1.0);
            }
        }
        let g = hg.clique_expand();
        let h = cheeger_constant_exact(&g);
        // K_6: balanced 3/3 split. cut = 3*3=9, vol(S)=3*5=15 each side.
        let expected = 9.0 / 15.0;
        assert!((h - expected).abs() < 1e-9, "K_6: h={h}, expected {expected}");
    }

    /// Cheeger's inequality must hold on a genuinely irregular, non-vertex-
    /// transitive graph (two triangles joined by a single bridge edge) —
    /// the interesting case, since symmetric graphs like cycles/complete
    /// graphs can trivially satisfy inequalities that a buggy implementation
    /// might also accidentally satisfy by symmetry.
    #[test]
    fn cheeger_inequality_holds_on_irregular_bridge_graph() {
        let mut hg = Hypergraph::new(6);
        for &(a, b) in &[(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3), (2, 3)] {
            hg.add_hyperedge(vec![a, b], 1.0);
        }
        let g = hg.clique_expand();
        let check = cheeger_inequality_check(&g);
        assert!(
            check.holds,
            "Cheeger's inequality violated: lambda_1={}, h={}, bounds=[{}, {}]",
            check.lambda_1, check.h, check.lower_bound, check.upper_bound
        );
        // The bridge edge (2,3) should be at or near the optimal cut,
        // since it's the unique edge whose removal disconnects the graph
        // into the two triangles -- a sanity check that h is small (a weak
        // expander) rather than some unrelated value.
        assert!(check.h < 0.5, "bridge graph should be a weak expander, got h={}", check.h);
    }

    /// A disconnected graph has lambda_1 = 0 and h = 0 (the empty cut
    /// between components is free) -- both sides of Cheeger's inequality
    /// degenerate to 0 = 0, a useful edge-case check.
    #[test]
    fn disconnected_graph_gives_zero_gap_and_zero_cheeger_constant() {
        let mut hg = Hypergraph::new(6);
        for &(a, b) in &[(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3)] {
            hg.add_hyperedge(vec![a, b], 1.0);
        }
        let g = hg.clique_expand();
        let check = cheeger_inequality_check(&g);
        assert!(check.lambda_1 < 1e-9, "lambda_1={}", check.lambda_1);
        assert!(check.h < 1e-9, "h={}", check.h);
        assert!(check.holds);
    }
}
