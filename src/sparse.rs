//! Matrix-free sparse representation of the symmetric normalized Laplacian.
//!
//! `laplacian.rs` builds L_sym = I - D^{-1/2} A D^{-1/2} as a dense N×N
//! `DMatrix`, which is fine up to a few thousand vertices but is an O(N^2)
//! memory / O(N^3) eigendecomposition wall at N = 10^4 (800 MB just to
//! store the matrix, and a dense `SymmetricEigen` call that pure-Rust
//! nalgebra will not finish in any reasonable time). The stochastic-trace
//! estimator in `spectral_trace.rs` only ever needs matrix-vector products
//! y = L_sym x, so we never materialize L_sym at all: this module stores
//! the graph as a weighted adjacency list (equivalent to CSR) and applies
//! the Laplacian directly as an operator: each output entry i is x_i minus
//! 1/sqrt(d_i) times the weighted sum, over neighbors j of i, of
//! w_ij * x_j / sqrt(d_j) -- in O(N + E) per matvec instead of O(N^2).

use crate::hypergraph::WeightedGraph;

pub struct SparseNormalizedLaplacian {
    pub n: usize,
    /// adj[i] = list of (neighbor, weight)
    adj: Vec<Vec<(usize, f64)>>,
    inv_sqrt_deg: Vec<f64>,
}

impl SparseNormalizedLaplacian {
    pub fn from_graph(g: &WeightedGraph) -> Self {
        let mut adj: Vec<Vec<(usize, f64)>> = vec![Vec::new(); g.n];
        let mut deg = vec![0.0_f64; g.n];
        for &((u, v), w) in &g.edges {
            adj[u].push((v, w));
            adj[v].push((u, w));
            deg[u] += w;
            deg[v] += w;
        }
        let inv_sqrt_deg = deg
            .iter()
            .map(|&d| if d > 1e-14 { 1.0 / d.sqrt() } else { 0.0 })
            .collect();
        Self {
            n: g.n,
            adj,
            inv_sqrt_deg,
        }
    }

    pub fn nnz(&self) -> usize {
        self.adj.iter().map(|l| l.len()).sum()
    }

    /// y = L_sym * x, matrix-free.
    pub fn matvec(&self, x: &[f64], y: &mut [f64]) {
        debug_assert_eq!(x.len(), self.n);
        debug_assert_eq!(y.len(), self.n);
        for i in 0..self.n {
            let mut acc = 0.0_f64;
            let inv_sqrt_di = self.inv_sqrt_deg[i];
            if inv_sqrt_di > 0.0 {
                for &(j, w) in &self.adj[i] {
                    acc += w * self.inv_sqrt_deg[j] * x[j];
                }
                acc *= inv_sqrt_di;
            }
            y[i] = x[i] - acc;
        }
    }
}

/// Lets `SparseNormalizedLaplacian` plug directly into `krylov_ds`'s
/// Arnoldi/Lanczos routines (see `spectral_trace.rs`), which only need a
/// matrix-vector product and the operator's dimension -- exactly what this
/// type already provides. This is a thin adapter, not new numerics: it
/// exists so the O(N+E) matvec above can be reused by a general,
/// independently-tested Krylov-subspace library instead of a bespoke
/// reimplementation of Lanczos.
impl krylov_ds::LinearOperator<f64> for SparseNormalizedLaplacian {
    fn dim(&self) -> usize {
        self.n
    }

    fn apply(&self, x: &[f64], y: &mut [f64]) {
        self.matvec(x, y);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hypergraph::WeightedGraph;
    use crate::laplacian::normalized_laplacian;

    #[test]
    fn matvec_matches_dense_laplacian() {
        // A small irregular weighted graph (not a nice symmetric shape,
        // deliberately, so a matvec bug can't hide behind symmetry).
        let g = WeightedGraph {
            n: 6,
            edges: vec![
                ((0, 1), 1.0),
                ((0, 2), 2.5),
                ((1, 2), 0.3),
                ((2, 3), 1.0),
                ((3, 4), 4.0),
                ((4, 5), 1.0),
                ((1, 5), 0.7),
            ],
        };
        let dense = normalized_laplacian(&g);
        let sparse = SparseNormalizedLaplacian::from_graph(&g);

        let mut rng_state: u64 = 12345;
        let mut next = || {
            // xorshift64, deterministic, no extra dependency needed for a unit test
            rng_state ^= rng_state << 13;
            rng_state ^= rng_state >> 7;
            rng_state ^= rng_state << 17;
            (rng_state as f64 / u64::MAX as f64) * 2.0 - 1.0
        };

        for _ in 0..5 {
            let x: Vec<f64> = (0..g.n).map(|_| next()).collect();
            let xv = nalgebra::DVector::from_vec(x.clone());
            let expected = &dense * &xv;

            let mut y = vec![0.0; g.n];
            sparse.matvec(&x, &mut y);

            for i in 0..g.n {
                assert!(
                    (y[i] - expected[i]).abs() < 1e-10,
                    "matvec mismatch at {i}: sparse={} dense={}",
                    y[i],
                    expected[i]
                );
            }
        }
    }
}
