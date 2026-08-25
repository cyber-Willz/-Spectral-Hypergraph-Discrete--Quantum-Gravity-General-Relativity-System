//! Distinction item #1: N ≥ 10^4 spectral dimension flow via matrix-free
//! stochastic Lanczos quadrature (SLQ), instead of dense diagonalization.
//!
//! Also includes a direct honesty check: run the *same* graph through both
//! the exact dense `laplacian::spectrum` + `heat_kernel` path (as large as
//! that remains tractable) and the SLQ path, and report the disagreement,
//! before trusting SLQ alone at N = 10^4 where the dense path no longer
//! finishes in reasonable time.

use spectral_dqg::continuum_limit::random_simple_regular_graph;
use spectral_dqg::heat_kernel::{heat_trace, spectral_dimension_flow};
use spectral_dqg::laplacian::spectrum;
use spectral_dqg::sparse::SparseNormalizedLaplacian;
use spectral_dqg::spectral_trace::spectral_dimension_flow_slq;
use std::time::Instant;

fn main() {
    let d = 4usize;

    // --- Step A: agreement check at a size dense diagonalization can still
    // finish, N = 2000, so we know SLQ's plateau is not a Lanczos artifact
    // before trusting it alone at N = 10^4. ---
    println!("=== Step A: SLQ vs exact dense diagonalization, N=2000, d={d} ===");
    let n_check = 2000usize;
    let g_check = random_simple_regular_graph(n_check, d, 1, 20000);

    let t0 = Instant::now();
    let exact = spectrum(&g_check, true);
    let exact_time = t0.elapsed();

    let sparse_check = SparseNormalizedLaplacian::from_graph(&g_check);
    let t0 = Instant::now();
    let flow_slq = spectral_dimension_flow_slq(&sparse_check, 1e-3, 30.0, 25, 40, 60, 11);
    let slq_time = t0.elapsed();
    let flow_exact = spectral_dimension_flow(&exact.eigenvalues, 1e-3, 30.0, 25);

    println!(
        "dense diagonalization: {:?}   SLQ (40 probes, 60 Lanczos steps): {:?}",
        exact_time, slq_time
    );
    println!("{:>12} {:>14} {:>14} {:>10}", "t", "d_s exact", "d_s SLQ", "|diff|");
    let mut max_diff = 0.0_f64;
    for (e, s) in flow_exact.iter().zip(flow_slq.iter()) {
        let diff = (e.d_s - s.d_s).abs();
        max_diff = max_diff.max(diff);
        println!("{:>12.4} {:>14.4} {:>14.4} {:>10.4}", e.t, e.d_s, s.d_s, diff);
    }
    println!("max |d_s exact - d_s SLQ| over sweep: {:.4}\n", max_diff);

    // --- Step B: the actual distinction-level result, N = 10,000, no dense
    // diagonalization attempted at all (matrix-free throughout). ---
    println!("=== Step B: N=10,000, d={d}, matrix-free SLQ only ===");
    let n_big = 10_000usize;
    let t0 = Instant::now();
    let g_big = random_simple_regular_graph(n_big, d, 2, 20000);
    let build_time = t0.elapsed();
    let sparse_big = SparseNormalizedLaplacian::from_graph(&g_big);
    println!(
        "graph build: {:?}, N={}, nnz(L)={}",
        build_time,
        sparse_big.n,
        sparse_big.nnz()
    );

    let t0 = Instant::now();
    // More probes/steps than the N=2000 check since we can no longer
    // cross-check against a dense answer at this size -- we spend the
    // compute we saved by not diagonalizing on tightening the SLQ estimate
    // instead.
    let flow_big = spectral_dimension_flow_slq(&sparse_big, 1e-3, 60.0, 40, 80, 80, 21);
    let slq_big_time = t0.elapsed();
    println!("SLQ sweep (80 probes, 80 Lanczos steps, 40 t-values): {:?}\n", slq_big_time);

    println!("{:>12} {:>14} {:>14}", "t", "P(t)", "d_s(t)");
    for pt in &flow_big {
        println!("{:>12.5} {:>14.4} {:>14.4}", pt.t, pt.p_t, pt.d_s);
    }

    // Sanity bookends that don't need any eigensolve at all: P(t->0) -> N,
    // reported here via the SLQ estimate at the smallest swept t, just to
    // confirm the estimator isn't secretly biased at the UV end.
    let p_small_t = flow_big.first().map(|p| p.p_t).unwrap_or(f64::NAN);
    println!(
        "\nUV sanity check: P(t={:.2e}) = {:.1} (should be close to N = {})",
        flow_big.first().map(|p| p.t).unwrap_or(f64::NAN),
        p_small_t,
        n_big
    );
    let _ = heat_trace; // silence unused import if bookends above change
}
