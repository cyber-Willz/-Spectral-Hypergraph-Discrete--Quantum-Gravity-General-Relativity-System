use spectral_dqg::continuum_limit::{count_components, random_simple_regular_graph};
use spectral_dqg::heat_kernel::heat_trace;
use spectral_dqg::hypergraph::Hypergraph;
use spectral_dqg::ihara_zeta::{analyze, cross_check};
use spectral_dqg::laplacian::spectrum;
use spectral_dqg::nonbacktracking::{count_closed_nbt_walks_bruteforce, hashimoto_matrix, trace_bk};

fn sample_hypergraph() -> Hypergraph {
    let mut hg = Hypergraph::new(8);
    hg.add_hyperedge(vec![0, 1], 1.0);
    hg.add_hyperedge(vec![1, 2], 1.0);
    hg.add_hyperedge(vec![2, 3], 1.0);
    hg.add_hyperedge(vec![3, 0], 1.0);
    hg.add_hyperedge(vec![0, 1, 2], 1.0);
    hg.add_hyperedge(vec![4, 5], 1.0);
    hg.add_hyperedge(vec![5, 6], 1.0);
    hg.add_hyperedge(vec![6, 7], 1.0);
    hg.add_hyperedge(vec![7, 4], 1.0);
    hg.add_hyperedge(vec![3, 4], 1.0);
    hg.add_hyperedge(vec![2, 5, 7], 1.0);
    hg
}

#[test]
fn laplacian_smallest_eigenvalue_is_zero_and_nonnegative_spectrum() {
    let g = sample_hypergraph().clique_expand();
    let spec = spectrum(&g, true);
    assert!((spec.eigenvalues[0]).abs() < 1e-8);
    for &lam in &spec.eigenvalues {
        assert!(lam >= -1e-9, "normalized Laplacian must be PSD, got {lam}");
        assert!(lam <= 2.0 + 1e-9, "normalized Laplacian eigenvalues are bounded by 2");
    }
}

#[test]
fn heat_trace_at_t_zero_equals_vertex_count() {
    let g = sample_hypergraph().clique_expand();
    let spec = spectrum(&g, true);
    let p0 = heat_trace(&spec.eigenvalues, 0.0);
    assert!((p0 - g.n as f64).abs() < 1e-9);
}

#[test]
fn heat_trace_is_monotonically_decreasing_in_t() {
    let g = sample_hypergraph().clique_expand();
    let spec = spectrum(&g, true);
    let ts = [0.01, 0.1, 1.0, 5.0, 20.0];
    let mut prev = f64::INFINITY;
    for &t in &ts {
        let p = heat_trace(&spec.eigenvalues, t);
        assert!(p <= prev + 1e-9);
        prev = p;
    }
}

#[test]
fn ihara_and_bass_formulas_agree() {
    let g = sample_hypergraph().clique_expand();
    let data = analyze(&g);
    let checks = cross_check(&g, &data, &[0.02, 0.05, 0.1, 0.15, 0.2, 0.25]);
    for (u, via_b, via_bass, err) in checks {
        assert!(err < 1e-8, "zeta formulas disagree at u={u}: {via_b} vs {via_bass}");
    }
}

#[test]
fn trace_bk_matches_bruteforce_closed_walk_count() {
    let g = sample_hypergraph().clique_expand();
    let (b, _) = hashimoto_matrix(&g);
    for k in 0..=6 {
        let tr = trace_bk(&b, k);
        let bf = count_closed_nbt_walks_bruteforce(&g, k) as f64;
        assert!(
            (tr - bf).abs() < 1e-6,
            "k={k}: Tr(B^k)={tr} but brute-force count={bf}"
        );
    }
}

#[test]
fn no_closed_nbt_walks_of_length_one_or_two() {
    // Structural sanity check independent of the graph: a length-1 closed
    // walk needs a self-loop (none in a simple graph), and a length-2
    // closed walk (u->v->u) is definitionally a backtrack.
    let g = sample_hypergraph().clique_expand();
    assert_eq!(count_closed_nbt_walks_bruteforce(&g, 1), 0);
    assert_eq!(count_closed_nbt_walks_bruteforce(&g, 2), 0);
}

#[test]
fn simple_regular_sampler_is_actually_simple_and_d_regular() {
    let d = 4;
    for &n in &[20usize, 100] {
        let g = random_simple_regular_graph(n, d, 7, 5000);
        assert_eq!(g.n, n);
        assert_eq!(g.edges.len(), n * d / 2, "wrong edge count for a simple d-regular graph");
        // No self-loops, no duplicate edges, every weight exactly 1.
        let mut seen = std::collections::HashSet::new();
        for &((a, b), w) in &g.edges {
            assert_ne!(a, b, "self-loop found: sampler is not producing a simple graph");
            assert!((w - 1.0).abs() < 1e-12, "multi-edge weight found: {w}");
            assert!(seen.insert((a.min(b), a.max(b))), "duplicate edge found");
        }
        for v in 0..n {
            assert!((g.degree(v) - d as f64).abs() < 1e-9, "vertex {v} is not exactly d-regular");
        }
        assert_eq!(count_components(&g), 1, "expected a connected sample");
    }
}
