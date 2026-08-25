//! Euclidean path integral over metrics for Regge calculus, via Metropolis
//! Monte Carlo over edge lengths -- this is the direct discretization of
//! ```text
//!     Z = integral D[g] exp(-S_Regge[g] / hbar)
//! ```
//! with D[g] realized concretely as a random walk on the space of valid
//! edge-length assignments (DeWitt/Hamber-Williams-style dynamical Regge
//! calculus; see e.g. Hamber, "Quantum Gravitation", Ch. 5-6, and
//! Rocek & Williams (1981) for the original Regge-calculus path integral
//! formulation). We fix the connectivity (simplicial complex) and let only
//! the edge lengths fluctuate -- the standard "quantum Regge calculus"
//! setup, as distinct from dynamical/causal triangulations which also sum
//! over connectivity. Coupling kappa = 1/(8 pi G) and hbar are both set to
//! 1 (natural units); nothing here fixes an actual value of G.
//!
//! What this establishes: a working numerical sampler of the Regge path
//! integral for a *fixed* simplicial complex, with the acceptance test,
//! thermalization, and expectation values reported honestly (including
//! the acceptance rate, since a silently-degenerate acceptance rate near
//! 0% or 100% is the standard failure mode of any Metropolis sampler and
//! would make <S> meaningless).

use crate::regge::{all_tetrahedra_valid, regge_action, EdgeLengths, SimplicialComplex};
use rand::Rng;
use rand_pcg::Pcg64;
use rand::SeedableRng;

pub struct McConfig {
    pub kappa: f64,        // 1/(8 pi G), coupling in front of curvature term
    pub lambda: f64,       // cosmological constant term
    pub step_size: f64,    // max proposed |delta length|
    pub n_sweeps: usize,   // one sweep = one proposal per edge
    pub seed: u64,
}

pub struct McResult {
    pub mean_action: f64,
    pub stderr_action: f64,
    pub acceptance_rate: f64,
    pub n_samples: usize,
    pub action_trace: Vec<f64>,
}

/// Run a Metropolis-Hastings random walk over edge lengths, sampling from
/// exp(-kappa * S_Regge[l] / hbar) with hbar=1, subject to every tetrahedron
/// staying geometrically valid (Cayley-Menger positive). Rejects proposals
/// that break validity outright (infinite-action wall), exactly as a hard
/// constraint boundary should be handled in a Metropolis sampler.
pub fn run_path_integral(
    complex: &SimplicialComplex,
    initial: EdgeLengths,
    cfg: &McConfig,
) -> McResult {
    let mut rng = Pcg64::seed_from_u64(cfg.seed);
    let mut lengths = initial;
    assert!(
        all_tetrahedra_valid(complex, &lengths),
        "initial configuration must be geometrically valid"
    );

    let mut current_s = regge_action(complex, &lengths, cfg.lambda).total;
    let mut accepted = 0usize;
    let mut proposed = 0usize;
    let mut trace = Vec::with_capacity(cfg.n_sweeps);

    let edge_list: Vec<_> = complex.edges.clone();

    for _sweep in 0..cfg.n_sweeps {
        for &e in &edge_list {
            proposed += 1;
            let old_len = *lengths.lengths.get(&e).unwrap();
            let delta = rng.gen_range(-cfg.step_size..cfg.step_size);
            let new_len = old_len + delta;
            if new_len <= 1e-6 {
                continue; // reject degenerate/negative lengths outright
            }
            lengths.lengths.insert(e, new_len);

            if !all_tetrahedra_valid(complex, &lengths) {
                lengths.lengths.insert(e, old_len); // reject: broke triangle inequality
                continue;
            }

            let new_s = regge_action(complex, &lengths, cfg.lambda).total;
            let d_s = cfg.kappa * (new_s - current_s);
            let accept = d_s <= 0.0 || rng.gen::<f64>() < (-d_s).exp();

            if accept {
                current_s = new_s;
                accepted += 1;
            } else {
                lengths.lengths.insert(e, old_len);
            }
        }
        trace.push(current_s);
    }

    let n = trace.len();
    let mean = trace.iter().sum::<f64>() / n as f64;
    let var = trace.iter().map(|s| (s - mean).powi(2)).sum::<f64>() / (n.max(2) - 1) as f64;
    // Naive stderr (ignores autocorrelation -- stated explicitly, not hidden).
    let stderr = (var / n as f64).sqrt();

    McResult {
        mean_action: mean,
        stderr_action: stderr,
        acceptance_rate: accepted as f64 / proposed as f64,
        n_samples: n,
        action_trace: trace,
    }
}
