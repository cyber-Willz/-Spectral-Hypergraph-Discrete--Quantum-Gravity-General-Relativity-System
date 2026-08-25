//! Matrix-free estimation of the heat trace P(t) = Tr(e^{-tL}) for N large
//! enough that dense diagonalization (`laplacian::spectrum`, O(N^3)) is not
//! an option — this is what makes an N ≥ 10^4 `d_s(t)` sweep tractable.
//!
//! Method: Stochastic Lanczos Quadrature (Hutchinson trace estimator +
//! Gauss quadrature via the Lanczos tridiagonalization), following Ubaru,
//! Chen & Saad, "Fast Estimation of tr(f(A)) via Stochastic Lanczos
//! Quadrature", SIMAX 2017. For each of `n_probes` independent Rademacher
//! vectors v (entries ±1):
//!
//!   1. Run m-step Lanczos on L starting from v/||v||, producing the
//!      tridiagonal T_m (diagonal α, off-diagonal β) -- only matvecs, no L
//!      ever materialized. The three-term recurrence + reorthogonalization
//!      itself is delegated to `krylov_ds::Lanczos` (full
//!      reorthogonalization), an independently-tested general-purpose
//!      Krylov-subspace crate, rather than hand-rolled here: this module
//!      only implements `krylov_ds::LinearOperator` for
//!      `SparseNormalizedLaplacian` (in `sparse.rs`) and the SLQ-specific
//!      logic (probe sampling, Gauss quadrature, the t-sweep reuse below).
//!   2. Eigendecompose the small (m×m) T_m = Y Θ Y^T.
//!   3. v^T f(L) v ≈ ||v||^2 · Σ_i (Y[0,i])^2 f(θ_i)   (Gauss quadrature
//!      nodes θ_i, weights (Y[0,i])^2).
//!
//! Averaging Γ_k = v_k^T f(L) v_k over probes gives an unbiased estimator
//! of Tr(f(L)) since E[v v^T] = I for Rademacher v. We use full
//! reorthogonalization in the Lanczos loop (m is small -- a few dozen
//! steps -- so this costs O(m^2 N), negligible next to the O(m·nnz)
//! matvecs) because losing orthogonality silently manufactures spurious
//! duplicate Ritz values, which is exactly the kind of bug that would
//! produce a plausible-looking but wrong d_s(t) curve.

use crate::sparse::SparseNormalizedLaplacian;
use krylov_ds::{Lanczos, LinearOperator, Reorthogonalization};
use nalgebra::{DMatrix, SymmetricEigen};
use rand::{Rng, SeedableRng};
use rand_pcg::Pcg64;

/// One m-step Lanczos run from a fixed start vector, delegating the actual
/// three-term recurrence + reorthogonalization to `krylov_ds::Lanczos`
/// (full reorthogonalization -- the same policy the hand-rolled version
/// this replaced used, for the same reason: losing orthogonality silently
/// manufactures spurious duplicate Ritz values). `krylov_ds` normalizes
/// `start` internally, so it need not be pre-normalized here.
///
/// Returns `(alpha, beta)` with `alpha.len() == m` (or fewer, on happy
/// breakdown) and `beta.len() == alpha.len() - 1`, the standard tridiagonal
/// convention `quadrature_nodes_weights` below expects: `krylov_ds` reports
/// one extra trailing `beta` entry when it completes the full requested
/// depth without breakdown (a residual-bound quantity, not part of the
/// tridiagonal projection itself, per its own docs), which is dropped here.
fn lanczos_tridiagonal(
    l: &SparseNormalizedLaplacian,
    start: &[f64],
    m: usize,
) -> (Vec<f64>, Vec<f64>) {
    // krylov_ds errors if max_dim > n rather than silently clamping --
    // clamp here so callers (e.g. an SLQ sweep with a fixed step budget
    // run against graphs of varying size) don't have to special-case small
    // graphs themselves.
    let max_dim = m.min(l.dim()).max(1);
    let result = Lanczos::new(max_dim, 1e-12, Reorthogonalization::Full)
        .run(l, start)
        .expect("Lanczos on a Rademacher probe vector should not hit a dimension/zero-vector error");
    let alpha = result.alpha;
    let beta_len = alpha.len().saturating_sub(1);
    let beta = result.beta[..beta_len.min(result.beta.len())].to_vec();
    (alpha, beta)
}

/// Quadrature nodes θ_i and weights (Y[0,i])^2 from a tridiagonal (α, β).
fn quadrature_nodes_weights(alpha: &[f64], beta: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let m = alpha.len();
    let mut t = DMatrix::<f64>::zeros(m, m);
    for i in 0..m {
        t[(i, i)] = alpha[i];
    }
    for i in 0..beta.len() {
        t[(i, i + 1)] = beta[i];
        t[(i + 1, i)] = beta[i];
    }
    let eig = SymmetricEigen::new(t);
    let nodes: Vec<f64> = eig.eigenvalues.iter().cloned().collect();
    let weights: Vec<f64> = (0..m).map(|i| eig.eigenvectors[(0, i)].powi(2)).collect();
    (nodes, weights)
}

/// Stochastic-Lanczos-Quadrature estimate of Tr(e^{-tL}) at a single t,
/// reusing the same Lanczos runs (recomputed for each t here for clarity;
/// see `heat_trace_flow_slq` for the version that reuses one Lanczos run
/// across an entire t-sweep, which is what you actually want in practice).
pub fn heat_trace_slq(
    l: &SparseNormalizedLaplacian,
    t: f64,
    n_probes: usize,
    lanczos_steps: usize,
    seed: u64,
) -> f64 {
    let mut rng = Pcg64::seed_from_u64(seed);
    let n = l.n;
    let mut total = 0.0;
    for _ in 0..n_probes {
        let v: Vec<f64> = (0..n)
            .map(|_| if rng.gen_bool(0.5) { 1.0 } else { -1.0 })
            .collect();
        let (alpha, beta) = lanczos_tridiagonal(l, &v, lanczos_steps);
        let (nodes, weights) = quadrature_nodes_weights(&alpha, &beta);
        let v_norm_sq = n as f64; // ||v||^2 = n exactly for Rademacher entries
        let gamma: f64 = nodes
            .iter()
            .zip(&weights)
            .map(|(&theta, &w)| w * (-t * theta).exp())
            .sum::<f64>()
            * v_norm_sq;
        total += gamma;
    }
    total / n_probes as f64
}

/// The efficient version: run the Lanczos recursion (the O(m·nnz) part)
/// exactly once per probe vector, then reuse its quadrature nodes/weights
/// across the *entire* t-sweep (the exp() evaluation is O(m) per t, so this
/// is essentially free). This is what `spectral_dimension_flow_slq` uses.
pub fn heat_trace_flow_slq(
    l: &SparseNormalizedLaplacian,
    ts: &[f64],
    n_probes: usize,
    lanczos_steps: usize,
    seed: u64,
) -> Vec<f64> {
    let mut rng = Pcg64::seed_from_u64(seed);
    let n = l.n;
    let mut sums = vec![0.0_f64; ts.len()];
    for _ in 0..n_probes {
        let v: Vec<f64> = (0..n)
            .map(|_| if rng.gen_bool(0.5) { 1.0 } else { -1.0 })
            .collect();
        let (alpha, beta) = lanczos_tridiagonal(l, &v, lanczos_steps);
        let (nodes, weights) = quadrature_nodes_weights(&alpha, &beta);
        let v_norm_sq = n as f64;
        for (k, &t) in ts.iter().enumerate() {
            let gamma: f64 = nodes
                .iter()
                .zip(&weights)
                .map(|(&theta, &w)| w * (-t * theta).exp())
                .sum::<f64>()
                * v_norm_sq;
            sums[k] += gamma;
        }
    }
    sums.iter().map(|&s| s / n_probes as f64).collect()
}

/// d_s(t) via centered log-log finite difference, fed by SLQ-estimated
/// P(t) instead of an exact eigendecomposition. Mirrors
/// `heat_kernel::spectral_dimension_flow`'s estimator exactly so the two
/// are numerically comparable on graphs small enough to run both.
pub fn spectral_dimension_flow_slq(
    l: &SparseNormalizedLaplacian,
    t_min: f64,
    t_max: f64,
    n_samples: usize,
    n_probes: usize,
    lanczos_steps: usize,
    seed: u64,
) -> Vec<crate::heat_kernel::SpectralDimensionPoint> {
    assert!(n_samples >= 3);
    let log_min = t_min.ln();
    let log_max = t_max.ln();
    let ts: Vec<f64> = (0..n_samples)
        .map(|i| {
            let frac = i as f64 / (n_samples as f64 - 1.0);
            (log_min + frac * (log_max - log_min)).exp()
        })
        .collect();
    let ps = heat_trace_flow_slq(l, &ts, n_probes, lanczos_steps, seed);

    let mut out = Vec::with_capacity(n_samples - 2);
    for i in 1..n_samples - 1 {
        let d_ln_p = ps[i + 1].ln() - ps[i - 1].ln();
        let d_ln_t = ts[i + 1].ln() - ts[i - 1].ln();
        let d_s = -2.0 * d_ln_p / d_ln_t;
        out.push(crate::heat_kernel::SpectralDimensionPoint {
            t: ts[i],
            p_t: ps[i],
            d_s,
        });
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::heat_kernel::heat_trace as exact_heat_trace;
    use crate::hypergraph::Hypergraph;
    use crate::laplacian::spectrum;

    /// Cross-check: on a graph small enough to also diagonalize exactly,
    /// SLQ's estimate of P(t) must agree with the exact Σ e^{-tλ} to a few
    /// percent. This is the same "don't trust a new numerical method
    /// without checking it against a known-good one on a case you can
    /// afford to brute-force" discipline the rest of this crate uses.
    #[test]
    fn slq_matches_exact_heat_trace_on_small_graph() {
        // A moderately irregular graph: two triangles joined by a bridge,
        // plus a couple of extra chords, N = 12.
        let mut hg = Hypergraph::new(12);
        let triangle_edges = [
            (0, 1), (1, 2), (2, 0),
            (3, 4), (4, 5), (5, 3),
            (6, 7), (7, 8), (8, 6),
            (9, 10), (10, 11), (11, 9),
        ];
        for &(a, b) in &triangle_edges {
            hg.add_hyperedge(vec![a, b], 1.0);
        }
        // bridges connecting the four triangles into one component, plus
        // a couple of chords to break any residual symmetry
        for &(a, b) in &[(2, 3), (5, 6), (8, 9), (0, 7), (4, 10)] {
            hg.add_hyperedge(vec![a, b], 1.0);
        }
        let g = hg.clique_expand();

        let exact = spectrum(&g, true);
        let sparse_l = SparseNormalizedLaplacian::from_graph(&g);

        for &t in &[0.05, 0.3, 1.0, 3.0, 10.0] {
            let p_exact = exact_heat_trace(&exact.eigenvalues, t);
            // generous probe/step budget since N=12 is tiny -- the point
            // here is correctness of the method, not its N=10^4 economy
            let p_slq = heat_trace_slq(&sparse_l, t, 200, 12, 7);
            let rel_err = (p_exact - p_slq).abs() / p_exact;
            assert!(
                rel_err < 0.03,
                "t={t}: exact P(t)={p_exact}, SLQ P(t)={p_slq}, rel_err={rel_err}"
            );
        }
    }

    #[test]
    fn lanczos_steps_ge_n_recovers_exact_trace_deterministically() {
        // If lanczos_steps >= N, the Krylov space spans all of R^N (barring
        // degeneracy), so a *single* Rademacher probe already gives an
        // essentially exact quadrature for a generic vector -- this
        // isolates *quadrature exactness* from *Hutchinson sampling
        // variance*: with a full-rank Krylov space each individual probe's
        // v^T f(L) v is exact, but averaging finitely many Rademacher
        // probes still carries genuine Monte Carlo variance (Var[v^T A v]
        // = 2 sum_{i != j} A_ij^2 for Rademacher v), which is why the
        // tolerance below isn't machine epsilon even though the
        // quadrature step itself is exact here.
        let mut hg = Hypergraph::new(9);
        for &(a, b) in &[(0,1),(1,2),(2,0),(2,3),(3,4),(4,5),(5,3),(5,6),(6,7),(7,8),(8,6)] {
            hg.add_hyperedge(vec![a, b], 1.0);
        }
        let g = hg.clique_expand();
        let exact = spectrum(&g, true);
        let sparse_l = SparseNormalizedLaplacian::from_graph(&g);

        let t = 1.0;
        let p_exact = exact_heat_trace(&exact.eigenvalues, t);
        let p_slq = heat_trace_slq(&sparse_l, t, 4000, 9, 3);
        let rel_err = (p_exact - p_slq).abs() / p_exact;
        assert!(rel_err < 0.02, "rel_err={rel_err}");
    }
}
