//! Step 3 of the write-up ("continuum limit") asks for something that is
//! not, honestly, a runnable computation: proving that a *specific* sequence
//! of hypergraphs H_N converges to a *specific* smooth hyperbolic manifold
//! M = Γ\ℍ^D with a matching classical Selberg zeta function Z_S(s) is a
//! hard open research problem (it's essentially the discrete-to-continuum
//! problem in causal dynamical triangulations / causal set theory, unsolved
//! in general). Anything claiming to "just compute" that would be decorative,
//! not real.
//!
//! What *is* rigorously computable, and genuinely analogous, is the
//! well-established fact that random d-regular graphs converge (as N → ∞)
//! in empirical spectral distribution to the **Kesten–McKay law** — the
//! graph-theoretic analogue of the semicircle law, and the actual "N → ∞
//! expander limit" referenced in the write-up's Step 3 preamble. We compute
//! that convergence directly and report it, plus a discrete Ramanujan/
//! Alon–Boppana spectral-gap diagnostic (how close the non-backtracking
//! spectrum sits to the theoretical bound), which is the standard proxy in
//! the literature for "how expander-like / how close to the trace-formula
//! regime" a finite graph is. This is the strongest thing that can honestly
//! be said computationally about "closeness to a smooth limit" without a
//! specific target manifold to converge to.

use crate::hypergraph::WeightedGraph;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rand_pcg::Pcg64;

/// Attempt one configuration-model pairing; returns `None` if it produced a
/// self-loop or multi-edge (i.e. wasn't a simple graph), rather than
/// silently dropping/merging as the earlier generator did.
fn try_pairing(n: usize, d: usize, seed: u64) -> Option<WeightedGraph> {
    let mut rng = Pcg64::seed_from_u64(seed);
    let mut stubs: Vec<usize> = (0..n).flat_map(|v| std::iter::repeat(v).take(d)).collect();
    stubs.shuffle(&mut rng);

    let mut seen = std::collections::HashSet::new();
    let mut edges = Vec::with_capacity(n * d / 2);
    for pair in stubs.chunks(2) {
        let (mut a, mut b) = (pair[0], pair[1]);
        if a == b {
            return None; // self-loop -> reject this pairing entirely
        }
        if a > b {
            std::mem::swap(&mut a, &mut b);
        }
        if !seen.insert((a, b)) {
            return None; // multi-edge -> reject
        }
        edges.push(((a, b), 1.0));
    }
    Some(WeightedGraph { n, edges })
}

/// Rejection-sampled **simple** d-regular graph: retries the pairing model
/// with fresh randomness until it lands on a genuinely simple graph (no
/// self-loops, no multi-edges), rather than silently dropping/merging edges.
/// This is the standard configuration-model recipe (used by e.g. NetworkX's
/// `random_regular_graph`) and is what makes the Kesten-McKay / Ramanujan
/// comparison actually apply to the object being sampled. The rejection
/// probability for a random pairing to be simple is bounded below by a
/// constant depending only on d (not on n) for fixed d, so a modest retry
/// budget is enough in practice; if it's exhausted we say so explicitly
/// rather than quietly falling back to a non-simple graph.
pub fn random_simple_regular_graph(n: usize, d: usize, seed: u64, max_attempts: u64) -> WeightedGraph {
    assert!((n * d) % 2 == 0, "n*d must be even for a d-regular graph");
    for attempt in 0..max_attempts {
        if let Some(g) = try_pairing(n, d, seed.wrapping_add(attempt)) {
            return g;
        }
    }
    panic!(
        "random_simple_regular_graph: failed to sample a simple {d}-regular graph on {n} \
         vertices in {max_attempts} attempts"
    );
}

/// Configuration-model random d-regular multigraph on n vertices
/// (n*d must be even). Simple pairing model — may contain self-loops /
/// multi-edges for small n; kept around (only) to demonstrate the contrast
/// with `random_simple_regular_graph` below, see `main.rs`.
pub fn random_regular_graph(n: usize, d: usize, seed: u64) -> WeightedGraph {
    assert!((n * d) % 2 == 0, "n*d must be even for a d-regular graph");
    let mut rng = Pcg64::seed_from_u64(seed);
    let mut stubs: Vec<usize> = (0..n).flat_map(|v| std::iter::repeat(v).take(d)).collect();
    stubs.shuffle(&mut rng);

    let mut edge_weight = std::collections::HashMap::new();
    for pair in stubs.chunks(2) {
        if pair.len() < 2 {
            continue;
        }
        let (mut a, mut b) = (pair[0], pair[1]);
        if a == b {
            continue; // drop self-loops for the demo
        }
        if a > b {
            std::mem::swap(&mut a, &mut b);
        }
        *edge_weight.entry((a, b)).or_insert(0.0) += 1.0;
    }
    WeightedGraph {
        n,
        edges: edge_weight.into_iter().collect(),
    }
}

/// Empirical spectral density of the (simple, unweighted) adjacency matrix,
/// binned into `bins` buckets over [-2√(d-1), 2√(d-1)] — the Kesten–McKay
/// support for a d-regular tree/graph.
pub fn empirical_spectral_density(g: &WeightedGraph, d: usize, bins: usize) -> Vec<(f64, f64)> {
    use nalgebra::SymmetricEigen;
    let a = g.dense_adjacency();
    let eig = SymmetricEigen::new(a);
    let lambdas: Vec<f64> = eig.eigenvalues.iter().cloned().collect();
    let bound = 2.0 * ((d - 1) as f64).sqrt();
    let mut hist = vec![0.0; bins];
    let width = 2.0 * bound / bins as f64;
    for &lam in &lambdas {
        if lam < -bound || lam > bound {
            continue; // outside Kesten-McKay support (rare finite-size effect)
        }
        let idx = (((lam + bound) / width) as usize).min(bins - 1);
        hist[idx] += 1.0;
    }
    let norm = lambdas.len() as f64 * width;
    (0..bins)
        .map(|i| {
            let center = -bound + width * (i as f64 + 0.5);
            (center, hist[i] / norm)
        })
        .collect()
}

/// Kesten–McKay density at x for degree d, for direct comparison against
/// the empirical histogram above.
pub fn kesten_mckay_density(x: f64, d: usize) -> f64 {
    let d = d as f64;
    let bound = 2.0 * (d - 1.0).sqrt();
    if x.abs() >= bound {
        return 0.0;
    }
    (d / (2.0 * std::f64::consts::PI)) * (bound * bound - x * x).sqrt() / (d * d - x * x)
}

/// Number of connected components, via BFS. The configuration-model
/// generator above drops self-loops without rewiring, which both breaks
/// exact d-regularity for the affected vertex and can (rarely, but
/// non-negligibly at small N) leave the graph disconnected. The Ramanujan/
/// Alon-Boppana bound is stated for *connected* regular graphs; a
/// disconnected graph gets one extra eigenvalue near the trivial value per
/// extra component, which would otherwise look like an unexplained
/// near-violation of the bound. Reporting this directly is more honest than
/// silently assuming connectivity.
pub fn count_components(g: &WeightedGraph) -> usize {
    let adj = g.neighbors();
    let mut visited = vec![false; g.n];
    let mut components = 0;
    for start in 0..g.n {
        if visited[start] {
            continue;
        }
        components += 1;
        let mut stack = vec![start];
        visited[start] = true;
        while let Some(v) = stack.pop() {
            for &w in &adj[v] {
                if !visited[w] {
                    visited[w] = true;
                    stack.push(w);
                }
            }
        }
    }
    components
}

pub struct RamanujanDiagnostic {
    pub d: usize,
    pub alon_boppana_bound: f64,
    pub max_nontrivial_abs_eigenvalue: f64,
    pub fraction_within_bound: f64,
}

/// How close the graph's **adjacency** spectrum sits to the Ramanujan
/// (Alon–Boppana) bound |λ| ≤ 2√(d-1) for non-trivial eigenvalues, excluding
/// the trivial λ = d (Perron eigenvalue of a d-regular graph). This is the
/// standard finite proxy for "how expander-like", i.e. how close a finite
/// graph sits to the idealized trace-formula regime the write-up gestures
/// at with "dense expander-like structure".
///
/// Deliberately uses the *symmetric* adjacency matrix (robust, fast
/// `SymmetricEigen`) rather than the non-backtracking matrix B: the
/// equivalent Ihara-theoretic Ramanujan condition on B's spectrum is
/// mathematically equivalent for regular graphs, but nalgebra's pure-Rust
/// general (non-symmetric) `Schur` solver is unreliable and O(n^3) with a
/// bad constant on B (size 2·|E|) — using it here would trade correctness
/// and speed for no real benefit.
pub fn ramanujan_diagnostic(g: &WeightedGraph, d: usize) -> RamanujanDiagnostic {
    use nalgebra::SymmetricEigen;
    let a = g.dense_adjacency();
    let eig = SymmetricEigen::new(a);
    let bound = 2.0 * ((d - 1) as f64).sqrt();
    let trivial = d as f64;
    let nontrivial: Vec<f64> = eig
        .eigenvalues
        .iter()
        .cloned()
        .filter(|&lam| (lam - trivial).abs() > 1e-3)
        .collect();
    let max_nontrivial_abs_eigenvalue = nontrivial.iter().cloned().fold(0.0_f64, |m, x| m.max(x.abs()));
    let within = nontrivial.iter().filter(|&&lam| lam.abs() <= bound + 1e-6).count();
    let fraction_within_bound = if nontrivial.is_empty() {
        1.0
    } else {
        within as f64 / nontrivial.len() as f64
    };
    RamanujanDiagnostic {
        d,
        alon_boppana_bound: bound,
        max_nontrivial_abs_eigenvalue,
        fraction_within_bound,
    }
}
