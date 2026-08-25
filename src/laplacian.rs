//! Step 1: Discrete kinematics via the hypergraph Laplacian.
//!
//! Δ_H is built on the clique-expanded graph. We provide both the
//! unnormalized (D - A) and symmetric-normalized (I - D^{-1/2} A D^{-1/2})
//! forms; the normalized form is what makes the heat-kernel trace P(t) → 1
//! as t → ∞ regardless of vertex degree distribution, which is the
//! convention used in the spectral-dimension literature (Ambjørn–Jurkiewicz–
//! Loll, and the CDT / causal-set spectral-dimension estimators this
//! write-up is gesturing at).

use crate::hypergraph::WeightedGraph;
use nalgebra::{DMatrix, SymmetricEigen};

pub struct LaplacianSpectrum {
    /// Eigenvalues, ascending, λ_0 = 0 ≤ λ_1 ≤ ... ≤ λ_{N-1}.
    pub eigenvalues: Vec<f64>,
    pub normalized: bool,
}

pub fn unnormalized_laplacian(g: &WeightedGraph) -> DMatrix<f64> {
    g.dense_degree() - g.dense_adjacency()
}

/// Symmetric normalized Laplacian: L_sym = I - D^{-1/2} A D^{-1/2}.
/// Isolated vertices (degree 0) get a 0 row/col by convention.
pub fn normalized_laplacian(g: &WeightedGraph) -> DMatrix<f64> {
    let a = g.dense_adjacency();
    let d = g.dense_degree();
    let n = g.n;
    let mut d_inv_sqrt = DMatrix::<f64>::zeros(n, n);
    for i in 0..n {
        let di = d[(i, i)];
        if di > 1e-14 {
            d_inv_sqrt[(i, i)] = 1.0 / di.sqrt();
        }
    }
    let identity = DMatrix::<f64>::identity(n, n);
    identity - &d_inv_sqrt * &a * &d_inv_sqrt
}

pub fn spectrum(g: &WeightedGraph, normalized: bool) -> LaplacianSpectrum {
    let l = if normalized {
        normalized_laplacian(g)
    } else {
        unnormalized_laplacian(g)
    };
    let eig = SymmetricEigen::new(l);
    let mut eigenvalues: Vec<f64> = eig.eigenvalues.iter().cloned().collect();
    eigenvalues.sort_by(|a, b| a.partial_cmp(b).unwrap());
    // Clamp tiny negative numerical noise on the known-zero mode.
    if let Some(first) = eigenvalues.first_mut() {
        if first.abs() < 1e-9 {
            *first = 0.0;
        }
    }
    LaplacianSpectrum {
        eigenvalues,
        normalized,
    }
}
