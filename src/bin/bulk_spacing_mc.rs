//! Distinction item #2, full run: pool unfolded bulk nearest-neighbor
//! spacings of B's complex spectrum across many independent random-graph
//! seeds, and compare (via KS distance) against pooled Ginibre and Poisson
//! reference ensembles of matching size, built the same way.
//!
//! N is chosen deliberately small (see `schur_scale_probe`'s measured
//! timings: the Schur solve is O(m^3) with m = 2|E|, ~72s already at
//! N=800/d=4/m2=3200) so that 1000 independent seeds is something this
//! actually finishes, rather than a number quoted without having run it.

use spectral_dqg::bulk_spacing::{
    bulk_spacings_from_matrix, ks_distance, sample_poisson_disk, sample_real_ginibre,
};
use spectral_dqg::continuum_limit::random_simple_regular_graph;
use spectral_dqg::nonbacktracking::hashimoto_matrix;
use std::time::Instant;

fn main() {
    let n_vertices = 120usize;
    let d = 4usize;
    let n_seeds = 1000usize;
    let edge_frac = 0.25;
    let k_density = 5usize;

    let g0 = random_simple_regular_graph(n_vertices, d, 0, 20000);
    let (_b0, arcs0) = hashimoto_matrix(&g0);
    let m2 = arcs0.arcs.len();
    println!(
        "N={n_vertices} vertices, d={d} -> B is {m2}x{m2}. Running {n_seeds} seeds..."
    );

    let t_start = Instant::now();
    let mut pooled_b_spacings: Vec<f64> = Vec::new();
    let mut converged = 0usize;
    let mut failed = 0usize;

    for seed in 0..n_seeds as u64 {
        let g = if seed == 0 {
            g0.clone()
        } else {
            random_simple_regular_graph(n_vertices, d, seed, 20000)
        };
        let (b, _) = hashimoto_matrix(&g);
        match bulk_spacings_from_matrix(&b, edge_frac, k_density) {
            Some(res) => {
                pooled_b_spacings.extend(res.spacings);
                converged += 1;
            }
            None => failed += 1,
        }
        if (seed + 1) % 100 == 0 {
            eprintln!(
                "  ...{}/{n_seeds} seeds done, {:?} elapsed",
                seed + 1,
                t_start.elapsed()
            );
        }
    }
    let b_time = t_start.elapsed();
    println!(
        "B ensemble: {converged} converged, {failed} failed to converge, \
         {} pooled spacings, wall time {:?}",
        pooled_b_spacings.len(),
        b_time
    );

    // Reference ensembles, pooled over enough independent samples to be a
    // fair comparison to the pooled B spacings above (same matrix size m2,
    // so the reference solves cost about the same per-seed as the B ones).
    let n_ref_seeds = 200usize; // fewer needed: references have no
                                 // graph-construction variance to average
                                 // over, just the eigenvalue-repulsion
                                 // statistic itself
    let t_start = Instant::now();
    let mut pooled_ginibre: Vec<f64> = Vec::new();
    for seed in 0..n_ref_seeds as u64 {
        let m = sample_real_ginibre(m2, 10_000 + seed);
        if let Some(res) = bulk_spacings_from_matrix(&m, edge_frac, k_density) {
            pooled_ginibre.extend(res.spacings);
        }
    }
    println!(
        "Ginibre reference: {} pooled spacings, wall time {:?}",
        pooled_ginibre.len(),
        t_start.elapsed()
    );

    let t_start = Instant::now();
    let mut pooled_poisson: Vec<f64> = Vec::new();
    // matching bulk-radius scale: real Ginibre circular law has radius
    // ~sqrt(m2), use the same disk for the Poisson reference
    let radius = (m2 as f64).sqrt();
    for seed in 0..n_ref_seeds as u64 {
        let pts = sample_poisson_disk(m2, radius, 20_000 + seed);
        // reuse the same trim+unfold path via a thin wrapper: build spacings
        // directly since sample_poisson_disk already returns points, not a
        // matrix to diagonalize
        let bulk: Vec<_> = {
            use nalgebra::Complex;
            let mut radii: Vec<f64> = pts.iter().map(|z: &Complex<f64>| z.re.hypot(z.im)).collect();
            radii.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let cutoff = radii[((radii.len() as f64) * (1.0 - edge_frac)) as usize - 1];
            pts.into_iter().filter(|z| z.re.hypot(z.im) <= cutoff).collect::<Vec<_>>()
        };
        let n = bulk.len();
        if n > k_density + 2 {
            for i in 0..n {
                let mut dists: Vec<f64> = (0..n)
                    .filter(|&j| j != i)
                    .map(|j| (bulk[i].re - bulk[j].re).hypot(bulk[i].im - bulk[j].im))
                    .collect();
                dists.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let nn = dists[0];
                let kth = dists[k_density - 1];
                let local_density = k_density as f64 / (std::f64::consts::PI * kth * kth);
                pooled_poisson.push(nn * local_density.sqrt());
            }
        }
    }
    println!(
        "Poisson reference: {} pooled spacings, wall time {:?}\n",
        pooled_poisson.len(),
        t_start.elapsed()
    );

    let ks_b_vs_ginibre = ks_distance(&pooled_b_spacings, &pooled_ginibre);
    let ks_b_vs_poisson = ks_distance(&pooled_b_spacings, &pooled_poisson);
    let ks_ginibre_vs_poisson = ks_distance(&pooled_ginibre, &pooled_poisson);

    let frac_below = |s: &[f64], t: f64| s.iter().filter(|&&x| x < t).count() as f64 / s.len() as f64;
    println!("=== Results ===");
    println!(
        "fraction of unfolded spacings < 0.3 (small-spacing / repulsion proxy):"
    );
    println!("  B (non-backtracking):  {:.4}", frac_below(&pooled_b_spacings, 0.3));
    println!("  Ginibre reference:     {:.4}", frac_below(&pooled_ginibre, 0.3));
    println!("  Poisson reference:     {:.4}", frac_below(&pooled_poisson, 0.3));
    println!();
    println!("KS distance, B vs Ginibre reference:   {:.4}", ks_b_vs_ginibre);
    println!("KS distance, B vs Poisson reference:    {:.4}", ks_b_vs_poisson);
    println!("KS distance, Ginibre vs Poisson (sanity, should be large): {:.4}", ks_ginibre_vs_poisson);

    if ks_b_vs_ginibre < ks_b_vs_poisson {
        println!(
            "\n=> B's bulk spacing distribution sits closer to Ginibre than Poisson \
             (consistent with genuine eigenvalue repulsion / non-normal chaotic structure)."
        );
    } else {
        println!(
            "\n=> B's bulk spacing distribution sits closer to Poisson than Ginibre \
             at this N -- report this as the actual finding, not the expected one."
        );
    }
}
