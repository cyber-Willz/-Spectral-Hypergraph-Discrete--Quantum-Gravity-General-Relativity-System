use spectral_dqg::continuum_limit::random_simple_regular_graph;
use spectral_dqg::nonbacktracking::hashimoto_matrix;
use nalgebra::Schur;
use std::time::Instant;

fn main() {
    let d = 4usize;
    for &n in &[100usize, 200, 400, 800] {
        let g = random_simple_regular_graph(n, d, 99, 20000);
        let (b, arcs) = hashimoto_matrix(&g);
        let m2 = arcs.arcs.len();
        let t0 = Instant::now();
        let schur = Schur::try_new(b.clone(), 1e-10, 50_000);
        let elapsed = t0.elapsed();
        match schur {
            Some(s) => {
                let eigs = s.complex_eigenvalues();
                let trace_direct = b.trace();
                let sum_re: f64 = eigs.iter().map(|c| c.re).sum();
                println!(
                    "N={n} m2={m2} time={elapsed:?} converged=yes trace_diff={:.3e}",
                    (trace_direct - sum_re).abs()
                );
            }
            None => println!("N={n} m2={m2} time={elapsed:?} converged=NO"),
        }
    }
}
