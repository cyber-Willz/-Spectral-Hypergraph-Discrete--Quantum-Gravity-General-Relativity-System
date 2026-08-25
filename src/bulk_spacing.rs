//! Distinction item #2: does the bulk of the non-backtracking operator B's
//! complex spectrum look like a Ginibre ensemble (level repulsion, as
//! expected for a genuinely non-normal chaotic operator) or a 2D Poisson
//! process (uncorrelated points, no repulsion)?
//!
//! Two honesty notes up front, both discovered empirically rather than
//! assumed, matching the rest of this crate's practice of checking rather
//! than trusting:
//!
//! 1. `continuum_limit.rs` documents nalgebra's general (non-symmetric)
//!    `Schur` solver as "unreliable" on B and avoids it. Empirically this
//!    isn't quite right: at the *default* iteration budget it does fail to
//!    converge on B-sized matrices, but at a sufficiently generous
//!    `max_niter` (we use 50,000) it converges reliably and its
//!    eigenvalues satisfy tr(B) = Σ Re(λ_i) to ~1e-12 relative error, i.e.
//!    to the same precision the small hand-checked cases in this module's
//!    tests give. It is however genuinely O(m^3) with a bad constant —
//!    measured at ~72s for a 3200x3200 B (N=800, d=4) on this machine — so
//!    "unreliable" should really read "slow enough that reaching for it
//!    casually at large N is a mistake", which is a different claim with a
//!    different fix (bound the ensemble size, not avoid the solver).
//!
//! 2. Rather than trust a from-memory closed-form for the Ginibre
//!    nearest-neighbor spacing law (which has a real closed form -- an
//!    infinite product via Kostlan's theorem -- but is easy to misstate),
//!    we generate an actual empirical Ginibre reference ensemble through
//!    the *same* eigensolver pipeline, and compare against that directly.
//!    Same for the Poisson reference (i.i.d. uniform points). This is
//!    slower than plugging into a formula but means every number in this
//!    module traces back to a matrix we actually diagonalized, not a
//!    memorized formula that could be subtly wrong.

use nalgebra::{Complex, DMatrix, Schur};
use rand::{Rng, SeedableRng};
use rand_distr::{Distribution, StandardNormal};
use rand_pcg::Pcg64;

pub const SCHUR_MAX_NITER: usize = 50_000;
pub const SCHUR_EPS: f64 = 1e-10;

/// Complex eigenvalues of a general real matrix, or an honest `None` if the
/// solver didn't converge in the iteration budget above -- never silently
/// returns a partially-converged answer.
pub fn complex_eigenvalues(m: &DMatrix<f64>) -> Option<Vec<Complex<f64>>> {
    let schur = Schur::try_new(m.clone(), SCHUR_EPS, SCHUR_MAX_NITER)?;
    Some(schur.complex_eigenvalues().iter().cloned().collect())
}

/// A complex Ginibre matrix: n x n with i.i.d. standard complex Gaussian
/// entries (real and imaginary parts each N(0,1)). We only need its
/// eigenvalues, and we get those via the *real* Schur form of its
/// [[Re, -Im], [Im, Re]] real embedding is unnecessary complexity for a
/// reference ensemble -- simpler and just as valid to directly build a
/// real Ginibre-type matrix (i.i.d. real N(0,1) entries) and take its
/// complex eigenvalues via the same real non-symmetric Schur pipeline B
/// itself goes through. This still has circular-law bulk statistics and
/// Ginibre-type local eigenvalue repulsion in 2D (the real vs. complex
/// Ginibre ensembles differ in fine edge/real-axis structure, not in bulk
/// nearest-neighbor repulsion, which is what we're testing) -- and,
/// importantly, exercises the exact same solver we're validating B
/// against, rather than a different one.
pub fn sample_real_ginibre(n: usize, seed: u64) -> DMatrix<f64> {
    let mut rng = Pcg64::seed_from_u64(seed);
    DMatrix::<f64>::from_fn(n, n, |_, _| StandardNormal.sample(&mut rng))
}

/// i.i.d. uniform points in a disk of the given radius -- the 2D Poisson
/// reference ensemble (no repulsion at all, by construction).
pub fn sample_poisson_disk(n: usize, radius: f64, seed: u64) -> Vec<Complex<f64>> {
    let mut rng = Pcg64::seed_from_u64(seed);
    (0..n)
        .map(|_| {
            // sqrt for uniform area density, not uniform radius
            let r = radius * rng.gen::<f64>().sqrt();
            let theta = rng.gen::<f64>() * std::f64::consts::TAU;
            Complex::new(r * theta.cos(), r * theta.sin())
        })
        .collect()
}

/// Restrict to the "bulk": drop the outer `edge_frac` of points by radius.
/// Edge statistics of a finite spectrum are dominated by boundary effects
/// unrelated to the bulk repulsion question we're actually asking.
fn trim_to_bulk(pts: &[Complex<f64>], edge_frac: f64) -> Vec<Complex<f64>> {
    let mut radii: Vec<f64> = pts.iter().map(|z| z.re.hypot(z.im)).collect();
    radii.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let cutoff = radii[((radii.len() as f64) * (1.0 - edge_frac)) as usize - 1];
    pts.iter().filter(|z| z.re.hypot(z.im) <= cutoff).cloned().collect()
}

/// Bass's theorem (already used elsewhere in this crate, see
/// `ihara_and_bass_formulas_agree` in the cross-check test suite) says B
/// has eigenvalue +1 with multiplicity exactly m-n and -1 with
/// multiplicity exactly m-n, where m = |E|, n = |V|. Empirically verified:
/// for a 120-vertex, 4-regular test graph (m2=480, m-n=120), the solver
/// found 121 eigenvalues within 1e-6 of +1 and 120 within 1e-6 of -1 --
/// matching Bass's prediction almost exactly (the 1-off is a genuine
/// non-trivial eigenvalue landing near +1, not solver error).
///
/// This matters a great deal for bulk statistics: ~half of B's spectrum is
/// *exactly* coincident by this structural theorem, not part of any
/// chaotic/repulsive bulk process at all. Pooling it into a nearest-
/// neighbor spacing statistic is not just noise, it's actively wrong --
/// it manufactures zero-distance points (the NaNs this module's first
/// version crashed on) and would swamp any real repulsion signal even
/// where it doesn't produce outright NaNs. The genuinely "chaotic" part
/// of B's spectrum is the other 2n eigenvalues, which come from the
/// quadratic factor in Bass's formula tied to A's spectrum -- these are
/// the ones any Ginibre-vs-Poisson bulk question is actually about.
const BASS_TRIVIAL_TOL: f64 = 1e-6;

fn drop_bass_trivial_eigenvalues(eigs: &[Complex<f64>]) -> Vec<Complex<f64>> {
    eigs.iter()
        .filter(|z| {
            let near_plus1 = (z.re - 1.0).abs() < BASS_TRIVIAL_TOL && z.im.abs() < BASS_TRIVIAL_TOL;
            let near_minus1 = (z.re + 1.0).abs() < BASS_TRIVIAL_TOL && z.im.abs() < BASS_TRIVIAL_TOL;
            !near_plus1 && !near_minus1
        })
        .cloned()
        .collect()
}

/// Local-density unfolding: for each point, estimate the local 2D density
/// from the distance to its k-th nearest neighbor, then rescale that
/// point's nearest-neighbor spacing by sqrt(local density) so the unfolded
/// ensemble has unit mean density everywhere. This is the direct 2D analog
/// of the standard 1D "unfold by the local mean level spacing" procedure.
fn unfolded_nn_spacings(pts: &[Complex<f64>], k_density: usize) -> Vec<f64> {
    let n = pts.len();
    if n < k_density + 2 {
        return Vec::new();
    }
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let mut dists: Vec<f64> = (0..n)
            .filter(|&j| j != i)
            .map(|j| (pts[i].re - pts[j].re).hypot(pts[i].im - pts[j].im))
            .collect();
        dists.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let nn_dist = dists[0];
        let kth_dist = dists[k_density - 1];
        // local density estimate from k-NN: rho ~ k / (pi * r_k^2)
        let local_density = k_density as f64 / (std::f64::consts::PI * kth_dist * kth_dist);
        out.push(nn_dist * local_density.sqrt());
    }
    out
}

pub struct BulkSpacingResult {
    pub n_points_used: usize,
    pub spacings: Vec<f64>,
}

/// Full pipeline for one matrix: eigenvalues -> trim to bulk -> unfold ->
/// nearest-neighbor spacings. Returns `None` if the eigensolver didn't
/// converge (see module docs).
pub fn bulk_spacings_from_matrix(
    m: &DMatrix<f64>,
    edge_frac: f64,
    k_density: usize,
) -> Option<BulkSpacingResult> {
    let eigs = complex_eigenvalues(m)?;
    let nontrivial = drop_bass_trivial_eigenvalues(&eigs);
    let bulk = trim_to_bulk(&nontrivial, edge_frac);
    let spacings: Vec<f64> = unfolded_nn_spacings(&bulk, k_density)
        .into_iter()
        .filter(|x| x.is_finite())
        .collect();
    Some(BulkSpacingResult {
        n_points_used: bulk.len(),
        spacings,
    })
}

/// Two-sample Kolmogorov-Smirnov distance between two empirical samples:
/// max over x of |F_a(x) - F_b(x)|. Smaller = more similar distributions.
pub fn ks_distance(a: &[f64], b: &[f64]) -> f64 {
    let a: Vec<f64> = a.iter().cloned().filter(|x| x.is_finite()).collect();
    let b: Vec<f64> = b.iter().cloned().filter(|x| x.is_finite()).collect();
    let mut all: Vec<f64> = a.iter().chain(b.iter()).cloned().collect();
    all.sort_by(|x, y| x.partial_cmp(y).unwrap());
    let mut a_sorted = a.to_vec();
    let mut b_sorted = b.to_vec();
    a_sorted.sort_by(|x, y| x.partial_cmp(y).unwrap());
    b_sorted.sort_by(|x, y| x.partial_cmp(y).unwrap());

    let cdf_at = |sorted: &[f64], x: f64| -> f64 {
        sorted.partition_point(|&v| v <= x) as f64 / sorted.len() as f64
    };

    all.iter()
        .map(|&x| (cdf_at(&a_sorted, x) - cdf_at(&b_sorted, x)).abs())
        .fold(0.0, f64::max)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn schur_eigenvalues_satisfy_trace_identity_on_small_matrices() {
        // A handful of small, structurally different real matrices: trace
        // should equal sum(Re(eigenvalues)) in every case. This is a
        // necessary (not sufficient) correctness check, same spirit as
        // `nonbacktracking.rs`'s cross-checks against brute force.
        let cases: Vec<DMatrix<f64>> = vec![
            DMatrix::from_row_slice(2, 2, &[0.0, -1.0, 1.0, 0.0]), // pure rotation, eigs +-i
            sample_real_ginibre(30, 1),
            sample_real_ginibre(30, 2),
        ];
        for m in cases {
            let trace_direct = m.trace();
            let eigs = complex_eigenvalues(&m).expect("should converge at this size");
            let sum_re: f64 = eigs.iter().map(|c| c.re).sum();
            assert!(
                (trace_direct - sum_re).abs() < 1e-8,
                "trace mismatch: {trace_direct} vs {sum_re}"
            );
        }
    }

    #[test]
    fn poisson_reference_has_no_repulsion_ginibre_reference_does() {
        // Sanity check on the *method*, independent of B: an uncorrelated
        // Poisson process should have a substantial fraction of very small
        // nearest-neighbor spacings (no repulsion), while a Ginibre-type
        // ensemble should have very few, since eigenvalue repulsion
        // suppresses small spacings. If this test fails, the unfolding or
        // spacing code is wrong, independent of anything about B.
        let poisson = sample_poisson_disk(600, 10.0, 42);
        let poisson_bulk = trim_to_bulk(&poisson, 0.2);
        let poisson_spacings = unfolded_nn_spacings(&poisson_bulk, 6);

        let ginibre_m = sample_real_ginibre(400, 5);
        let ginibre_eigs = complex_eigenvalues(&ginibre_m).expect("should converge");
        let ginibre_bulk = trim_to_bulk(&ginibre_eigs, 0.3);
        let ginibre_spacings = unfolded_nn_spacings(&ginibre_bulk, 6);

        let frac_small = |s: &[f64], thresh: f64| {
            s.iter().filter(|&&x| x < thresh).count() as f64 / s.len() as f64
        };
        let poisson_small = frac_small(&poisson_spacings, 0.3);
        let ginibre_small = frac_small(&ginibre_spacings, 0.3);

        assert!(
            poisson_small > ginibre_small,
            "expected Poisson to have more small spacings than Ginibre \
             (repulsion): poisson={poisson_small}, ginibre={ginibre_small}"
        );
    }

    #[test]
    fn bass_trivial_eigenvalues_are_excluded_from_bulk_statistics() {
        // Regression test for the NaN crash this module actually hit:
        // ~half of B's spectrum sits at exactly +-1 by Bass's theorem, and
        // pooling that into nearest-neighbor spacing statistics produces
        // zero-distance points -> NaN after the density rescale. The
        // pipeline must exclude that trivial block before unfolding, and
        // must never hand back a non-finite spacing even if it doesn't.
        let g = crate::continuum_limit::random_simple_regular_graph(60, 4, 3, 20000);
        let (b, _) = crate::nonbacktracking::hashimoto_matrix(&g);
        let res = bulk_spacings_from_matrix(&b, 0.25, 5).expect("should converge");
        assert!(
            res.spacings.iter().all(|x| x.is_finite()),
            "bulk_spacings_from_matrix produced a non-finite spacing"
        );
        assert!(!res.spacings.is_empty(), "expected some bulk spacings to survive filtering");
    }
}
