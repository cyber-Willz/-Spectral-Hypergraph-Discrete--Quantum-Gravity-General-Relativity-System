//! Step 0: Hypergraph representation.
//!
//! A hypergraph H = (V, E) with V vertices and hyperedges e ⊆ V, |e| ≥ 2.
//! Everything downstream (Laplacian, non-backtracking operator, Ihara-Selberg
//! zeta function) is classically defined on ordinary graphs. The standard,
//! honest way to bridge that gap is the **clique expansion**: replace every
//! hyperedge of size k by a weighted k-clique among its members, with edge
//! weight w(e)/(k-1) so that the induced vertex degree matches the weighted
//! hypergraph degree (this is the Zhou–Huang–Schölkopf convention). This is
//! a *reduction*, not a claim that clique expansion is the unique or best
//! possible discretization — for genuine multi-way (non-pairwise) spectral
//! theory you'd want the tensor/simplicial Laplacians the write-up gestures
//! at, which don't have a single agreed-upon spectral theory yet. We are
//! explicit about that trade-off rather than pretending it away.

use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct Hypergraph {
    pub n_vertices: usize,
    /// Each hyperedge: (member vertex indices, weight)
    pub hyperedges: Vec<(Vec<usize>, f64)>,
}

impl Hypergraph {
    pub fn new(n_vertices: usize) -> Self {
        Self {
            n_vertices,
            hyperedges: Vec::new(),
        }
    }

    pub fn add_hyperedge(&mut self, members: Vec<usize>, weight: f64) {
        assert!(members.len() >= 2, "hyperedges must have at least 2 members");
        assert!(
            members.iter().all(|&v| v < self.n_vertices),
            "hyperedge references out-of-range vertex"
        );
        self.hyperedges.push((members, weight));
    }

    /// Weighted hypergraph degree of a vertex: sum of incident hyperedge weights.
    pub fn hyper_degree(&self, v: usize) -> f64 {
        self.hyperedges
            .iter()
            .filter(|(members, _)| members.contains(&v))
            .map(|(_, w)| w)
            .sum()
    }

    /// Clique expansion: for each hyperedge e of size k with weight w,
    /// add a weighted clique on its members with per-edge weight w/(k-1).
    /// Parallel edges from different hyperedges are summed.
    pub fn clique_expand(&self) -> WeightedGraph {
        let mut edge_weight: HashMap<(usize, usize), f64> = HashMap::new();
        for (members, w) in &self.hyperedges {
            let k = members.len();
            if k < 2 {
                continue;
            }
            let per_edge = w / (k as f64 - 1.0);
            for i in 0..members.len() {
                for j in (i + 1)..members.len() {
                    let (mut a, mut b) = (members[i], members[j]);
                    if a > b {
                        std::mem::swap(&mut a, &mut b);
                    }
                    *edge_weight.entry((a, b)).or_insert(0.0) += per_edge;
                }
            }
        }
        WeightedGraph {
            n: self.n_vertices,
            edges: edge_weight.into_iter().collect(),
        }
    }
}

/// A simple undirected weighted graph, the output of clique expansion and
/// the common substrate for the Laplacian and Ihara-zeta machinery.
#[derive(Debug, Clone)]
pub struct WeightedGraph {
    pub n: usize,
    pub edges: Vec<((usize, usize), f64)>,
}

impl WeightedGraph {
    pub fn degree(&self, v: usize) -> f64 {
        self.edges
            .iter()
            .filter(|((a, b), _)| *a == v || *b == v)
            .map(|(_, w)| w)
            .sum()
    }

    pub fn dense_adjacency(&self) -> nalgebra::DMatrix<f64> {
        let mut a = nalgebra::DMatrix::<f64>::zeros(self.n, self.n);
        for &((u, v), w) in &self.edges {
            a[(u, v)] += w;
            a[(v, u)] += w;
        }
        a
    }

    pub fn dense_degree(&self) -> nalgebra::DMatrix<f64> {
        let mut d = nalgebra::DMatrix::<f64>::zeros(self.n, self.n);
        for v in 0..self.n {
            d[(v, v)] = self.degree(v);
        }
        d
    }

    /// Neighbor list, needed for the non-backtracking (Hashimoto) operator.
    /// Multi-edges collapse to one combinatorial neighbor for the *unweighted*
    /// non-backtracking walk structure (Ihara theory is defined on simple
    /// graphs); weights still feed the Laplacian side separately.
    pub fn neighbors(&self) -> Vec<Vec<usize>> {
        let mut adj = vec![Vec::new(); self.n];
        for &((u, v), _) in &self.edges {
            if !adj[u].contains(&v) {
                adj[u].push(v);
            }
            if !adj[v].contains(&u) {
                adj[v].push(u);
            }
        }
        adj
    }
}
