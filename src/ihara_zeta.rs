//! Step 2: The (Ihara-)Selberg zeta function of the clique-expanded graph.
//!
//! Z_H(u) = Π_{[p] prime} (1 - u^{ℓ(p)})^{-1}
//!
//! is computed two *independent* ways so they can be checked against each
//! other rather than trusted on faith:
//!
//!  (A) Ihara's theorem: `Z_H(u)^{-1} = det(I - u B)`, where B is the
//!      Hashimoto non-backtracking edge matrix. Equivalently
//!      `Z_H(u)^{-1} = Π_i (1 - u μ_i)` over the (generally complex)
//!      eigenvalues μ_i of B.
//!
//!  (B) The Bass determinant formula, which reduces the 2m×2m eigenvalue
//!      problem to an n×n one:
//!
//! ```text
//!      Z_H(u)^{-1} = (1 - u^2)^{|E|-|V|} · det(I - u A + u^2 (D - I))
//! ```
//!
//! Both are exact identities for a finite connected graph with no degree-1
//! "dangling" vertices causing trivial factors; we evaluate both at a set of
//! real u inside the radius of convergence and report the discrepancy so
//! any bug shows up as a nonzero residual instead of silently producing a
//! plausible-looking number.

use crate::hypergraph::WeightedGraph;
use crate::nonbacktracking::hashimoto_matrix;
use nalgebra::{Complex, DMatrix, Schur};

pub struct IharaData {
    pub b_eigenvalues: Vec<Complex<f64>>,
    pub n_vertices: usize,
    pub n_edges: usize,
}

pub fn analyze(g: &WeightedGraph) -> IharaData {
    let (b, _arcs) = hashimoto_matrix(g);
    let schur = Schur::new(b);
    let b_eigenvalues = schur.complex_eigenvalues().iter().cloned().collect();
    IharaData {
        b_eigenvalues,
        n_vertices: g.n,
        n_edges: g.edges.len(),
    }
}

/// Method (A): Z_H(u)^{-1} via the eigenvalues of the non-backtracking matrix.
pub fn zeta_inverse_via_b(data: &IharaData, u: f64) -> Complex<f64> {
    data.b_eigenvalues
        .iter()
        .fold(Complex::new(1.0, 0.0), |acc, &mu| acc * (Complex::new(1.0, 0.0) - mu * u))
}

/// Method (B): Z_H(u)^{-1} via the Bass determinant formula (n×n, real).
pub fn zeta_inverse_via_bass(g: &WeightedGraph, u: f64) -> f64 {
    let a = g.dense_adjacency();
    // Unweighted degree for Ihara theory: use combinatorial (simple-graph) degree.
    let adj = g.neighbors();
    let n = g.n;
    let mut d = DMatrix::<f64>::zeros(n, n);
    let mut a_simple = DMatrix::<f64>::zeros(n, n);
    for v in 0..n {
        d[(v, v)] = adj[v].len() as f64;
        for &w in &adj[v] {
            a_simple[(v, w)] = 1.0;
        }
    }
    let _ = a; // weighted adjacency unused for Ihara (which is a simple-graph theory)
    let identity = DMatrix::<f64>::identity(n, n);
    let m = &identity - &a_simple * u + (&d - &identity) * (u * u);
    let det_term = m.determinant();
    let n_edges = adj.iter().map(|l| l.len()).sum::<usize>() / 2;
    let exponent = n_edges as i64 - n as i64;
    (1.0 - u * u).powi(exponent as i32) * det_term
}

/// Report the discrepancy between the two methods at a sweep of u values,
/// to make correctness (or a bug) visible rather than assumed.
pub fn cross_check(g: &WeightedGraph, data: &IharaData, u_values: &[f64]) -> Vec<(f64, f64, f64, f64)> {
    u_values
        .iter()
        .map(|&u| {
            let via_b = zeta_inverse_via_b(data, u).re;
            let via_bass = zeta_inverse_via_bass(g, u);
            let abs_err = (via_b - via_bass).abs();
            (u, via_b, via_bass, abs_err)
        })
        .collect()
}
