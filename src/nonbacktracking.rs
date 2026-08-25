//! The non-backtracking (Hashimoto) edge adjacency operator B.
//!
//! For a simple graph with m undirected edges, form the 2m directed arcs
//! (u→v) and (v→u) for each edge. B is the 2m×2m 0/1 matrix with
//! B[(u→v), (v→w)] = 1  iff  w ≠ u   (i.e. the walk continues at v without
//! immediately backtracking to u). Ihara's theorem says
//!
//! ```text
//!     Z_H(u)^{-1} = det(I - u B)
//! ```
//!
//! exactly, with no reduction — this is the cleanest computational route to
//! the zeta function and also the operator this user's `nbsc` crate
//! (non-backtracking spectral convolution) already builds for a different
//! purpose, so this module intentionally mirrors that construction.

use crate::hypergraph::WeightedGraph;
use nalgebra::DMatrix;

pub struct Arcs {
    /// arcs[i] = (u, v) meaning the directed step u -> v
    pub arcs: Vec<(usize, usize)>,
}

pub fn build_arcs(g: &WeightedGraph) -> Arcs {
    let adj = g.neighbors();
    let mut arcs = Vec::new();
    for u in 0..g.n {
        for &v in &adj[u] {
            arcs.push((u, v));
        }
    }
    Arcs { arcs }
}

/// Dense Hashimoto non-backtracking matrix B (size 2m x 2m).
pub fn hashimoto_matrix(g: &WeightedGraph) -> (DMatrix<f64>, Arcs) {
    let arcs = build_arcs(g);
    let m2 = arcs.arcs.len();
    let mut b = DMatrix::<f64>::zeros(m2, m2);
    for (i, &(u, v)) in arcs.arcs.iter().enumerate() {
        for (j, &(v2, w)) in arcs.arcs.iter().enumerate() {
            if v2 == v && w != u {
                b[(i, j)] = 1.0;
            }
        }
    }
    (b, arcs)
}

/// Trace of B^k, computed by direct dense matrix power. Used as one of the
/// three independent cross-checks on the zeta function coefficients.
pub fn trace_bk(b: &DMatrix<f64>, k: usize) -> f64 {
    if k == 0 {
        return b.nrows() as f64;
    }
    let mut acc = b.clone();
    for _ in 1..k {
        acc = &acc * b;
    }
    acc.trace()
}

/// Brute-force count of closed non-backtracking walks of length k starting
/// and ending anywhere (i.e. Σ over all closed non-backtracking walks),
/// found by direct graph traversal rather than linear algebra. This is the
/// third, purely combinatorial, cross-check: it should equal Tr(B^k).
pub fn count_closed_nbt_walks_bruteforce(g: &WeightedGraph, k: usize) -> u64 {
    let adj = g.neighbors();
    let mut count: u64 = 0;
    // A closed non-backtracking *arc* walk of length k corresponds to a
    // vertex sequence v0, v1, ..., vk = v0 with v_i ~ v_{i+1} and no
    // immediate backtrack at every internal step (v_{i-1} != v_{i+1}).
    // Crucially, since this must match Tr(B^k) -- a trace, i.e. a genuinely
    // *cyclic* quantity over arcs a_0,...,a_{k-1},a_0 -- the non-backtracking
    // condition also applies at the seam where the walk closes back into its
    // own first step: the arc (v_{k-1} -> v_0) must not immediately reverse
    // into the arc (v_0 -> v_1), i.e. we additionally require v1 != v_{k-1}.
    // Omitting that wrap-around check is a subtle but real bug (it silently
    // overcounts) -- caught here precisely because this function exists to
    // be cross-checked against Tr(B^k) rather than trusted in isolation.
    fn dfs(
        adj: &[Vec<usize>],
        start: usize,
        first: usize,
        prev: usize,
        cur: usize,
        steps_left: usize,
        count: &mut u64,
    ) {
        if steps_left == 0 {
            if cur == start && prev != first {
                *count += 1;
            }
            return;
        }
        for &next in &adj[cur] {
            if next == prev {
                continue; // no backtracking on internal steps
            }
            dfs(adj, start, first, cur, next, steps_left - 1, count);
        }
    }
    if k == 0 {
        // Trivial closed walk of length 0 at each of the 2m arcs.
        let m2: usize = adj.iter().map(|l| l.len()).sum();
        return m2 as u64;
    }
    for start in 0..g.n {
        for &first in &adj[start] {
            dfs(&adj, start, first, start, first, k - 1, &mut count);
        }
    }
    count
}
