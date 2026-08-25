//! Discrete quantum gravity: Regge calculus on a 3D simplicial complex.
//!
//! This module is deliberately scoped to what is actually well-defined and
//! computable, mirroring the honesty conventions of the rest of this crate
//! (see `continuum_limit.rs`). It implements:
//!
//!   1. A 3D simplicial complex (tetrahedra + their faces/edges), built from
//!      4-vertex hyperedges -- i.e. we now take "hyperedge of size 4" to
//!      literally mean "3-simplex", rather than clique-expanding it away.
//!   2. Regge's edge-length formulation of piecewise-flat geometry: no
//!      ambient embedding is assumed a priori. Edge lengths ARE the metric
//!      degrees of freedom (this is the actual content of Regge calculus --
//!      curvature is read off from edge lengths alone via the Cayley-Menger
//!      relations, exactly as angles in ordinary geometry are determined by
//!      side lengths alone).
//!   3. Deficit angles at hinges (edges, since this is 3D: hinges are
//!      (D-2)-simplices = 1-simplices) and the Regge action
//!      ```text
//!      S = sum_hinges  L_hinge * delta_hinge
//!      ```
//!      which is the exact discretization of the Einstein-Hilbert action
//!      integral(R sqrt(g) d^3x) for a piecewise-flat manifold (Regge 1961).
//!
//! What this module does NOT claim:
//!   - That this particular simplicial complex approximates any specific
//!     smooth manifold (same caveat as continuum_limit.rs Step 3).
//!   - Full continuum diffeomorphism invariance. Regge calculus is well
//!     known in the literature to only approximately restore diffeomorphism
//!     invariance in the continuum limit; at finite lattice spacing the
//!     only *exact* discrete residue is triangulation-independence of flat
//!     (zero-curvature) configurations -- that specific, narrower claim is
//!     what diffeomorphism_test.rs actually tests, and it is stated that
//!     way rather than as "diffeomorphism invariance" unqualified.

use std::collections::{HashMap, HashSet};

pub type VertexId = usize;
pub type Edge = (VertexId, VertexId);

fn edge_key(a: VertexId, b: VertexId) -> Edge {
    if a < b {
        (a, b)
    } else {
        (b, a)
    }
}

/// A 3-simplex (tetrahedron): 4 vertices, no embedding assumed.
#[derive(Debug, Clone, Copy)]
pub struct Tetrahedron {
    pub v: [VertexId; 4],
}

impl Tetrahedron {
    pub fn edges(&self) -> [Edge; 6] {
        let v = self.v;
        [
            edge_key(v[0], v[1]),
            edge_key(v[0], v[2]),
            edge_key(v[0], v[3]),
            edge_key(v[1], v[2]),
            edge_key(v[1], v[3]),
            edge_key(v[2], v[3]),
        ]
    }
}

/// The simplicial complex: a set of tetrahedra plus the derived edge set.
/// Built from 4-vertex hyperedges (see `hypergraph::Hypergraph`).
pub struct SimplicialComplex {
    pub tetrahedra: Vec<Tetrahedron>,
    pub edges: Vec<Edge>,
    /// edge -> indices into `tetrahedra` of every tet containing that edge.
    pub edge_to_tets: HashMap<Edge, Vec<usize>>,
}

impl SimplicialComplex {
    pub fn from_tetrahedra(tets: Vec<[VertexId; 4]>) -> Self {
        let tetrahedra: Vec<Tetrahedron> = tets.into_iter().map(|v| Tetrahedron { v }).collect();
        let mut edge_set: HashSet<Edge> = HashSet::new();
        let mut edge_to_tets: HashMap<Edge, Vec<usize>> = HashMap::new();
        for (ti, t) in tetrahedra.iter().enumerate() {
            for e in t.edges() {
                edge_set.insert(e);
                edge_to_tets.entry(e).or_default().push(ti);
            }
        }
        let mut edges: Vec<Edge> = edge_set.into_iter().collect();
        edges.sort();
        Self {
            tetrahedra,
            edges,
            edge_to_tets,
        }
    }

    /// An edge is "interior" (a genuine hinge with a well-defined deficit
    /// angle in the closed-manifold sense) if it is shared by tetrahedra
    /// whose dihedral angles are meant to close up around it. Boundary
    /// edges need a separate (extrinsic-curvature) treatment that this
    /// module does not implement; we simply report the incident-tet count
    /// so callers can decide, and restrict the action sum to hinges with
    /// count >= 3 (the minimum needed for the "gluing" picture to be
    /// non-degenerate) as an explicit, stated convention.
    pub fn hinge_multiplicity(&self, e: &Edge) -> usize {
        self.edge_to_tets.get(e).map(|v| v.len()).unwrap_or(0)
    }
}

/// Edge-length assignment: the actual dynamical variables of Regge calculus.
#[derive(Clone, Debug)]
pub struct EdgeLengths {
    pub lengths: HashMap<Edge, f64>,
}

impl EdgeLengths {
    pub fn get(&self, a: VertexId, b: VertexId) -> f64 {
        *self
            .lengths
            .get(&edge_key(a, b))
            .expect("missing edge length")
    }
}

/// Cayley-Menger determinant for a tetrahedron's 6 edge lengths. Its sign
/// (for the standard normalization below) determines geometric validity:
/// a positive value of `cm_volume_squared_times_288` means the four points
/// admit a genuine (non-degenerate) embedding in R^3 with these pairwise
/// distances -- i.e. the tetrahedron inequality is satisfied. This is the
/// *only* validity check Regge calculus needs; no ambient embedding of the
/// full complex is required, only local consistency per tetrahedron.
///
/// Returns 288 * Volume^2 (the standard integer-free scaling of the 5x5
/// Cayley-Menger determinant for a tetrahedron).
pub fn cayley_menger_288_vol2(lengths: &EdgeLengths, t: &Tetrahedron) -> f64 {
    let d = |i: usize, j: usize| -> f64 {
        let a = t.v[i];
        let b = t.v[j];
        lengths.get(a, b).powi(2)
    };
    // 5x5 Cayley-Menger matrix (Gram-like), see e.g. Blumenthal (1953).
    //     [0  1    1    1    1  ]
    //     [1  0   d01  d02  d03 ]
    // M = [1 d01   0   d12  d13 ]
    //     [1 d02  d12   0   d23 ]
    //     [1 d03  d13  d23   0  ]
    let m = nalgebra::Matrix5::new(
        0.0, 1.0, 1.0, 1.0, 1.0, //
        1.0, 0.0, d(0, 1), d(0, 2), d(0, 3), //
        1.0, d(0, 1), 0.0, d(1, 2), d(1, 3), //
        1.0, d(0, 2), d(1, 2), 0.0, d(2, 3), //
        1.0, d(0, 3), d(1, 3), d(2, 3), 0.0,
    );
    m.determinant()
}

pub fn tetrahedron_volume(lengths: &EdgeLengths, t: &Tetrahedron) -> f64 {
    let cm = cayley_menger_288_vol2(lengths, t);
    (cm.max(0.0) / 288.0).sqrt()
}

pub fn is_valid_tetrahedron(lengths: &EdgeLengths, t: &Tetrahedron) -> bool {
    cayley_menger_288_vol2(lengths, t) > 1e-12
}

/// Embed a single tetrahedron's 4 vertices in R^3 given only its 6 pairwise
/// edge lengths (always possible for a geometrically valid tetrahedron; the
/// embedding is unique up to a rigid motion, which is exactly the gauge
/// freedom Regge calculus is intrinsically insensitive to -- we only ever
/// use this local embedding to read off dihedral angles, never to assemble
/// a single global embedding of the whole complex).
fn embed_tetrahedron(lengths: &EdgeLengths, t: &Tetrahedron) -> Option<[nalgebra::Vector3<f64>; 4]> {
    use nalgebra::Vector3;
    let v = t.v;
    let l01 = lengths.get(v[0], v[1]);
    let l02 = lengths.get(v[0], v[2]);
    let l03 = lengths.get(v[0], v[3]);
    let l12 = lengths.get(v[1], v[2]);
    let l13 = lengths.get(v[1], v[3]);
    let l23 = lengths.get(v[2], v[3]);

    let p0 = Vector3::new(0.0, 0.0, 0.0);
    let p1 = Vector3::new(l01, 0.0, 0.0);

    // p2 in the xy-plane: |p2-p0|=l02, |p2-p1|=l12
    let x2 = (l01.powi(2) + l02.powi(2) - l12.powi(2)) / (2.0 * l01);
    let y2_sq = l02.powi(2) - x2.powi(2);
    if y2_sq < -1e-9 {
        return None;
    }
    let y2 = y2_sq.max(0.0).sqrt();
    let p2 = Vector3::new(x2, y2, 0.0);

    // p3 = (x3, y3, z3): |p3-p0| = l03, |p3-p1| = l13, |p3-p2| = l23.
    // From |p3-p0|^2 - |p3-p1|^2 = l03^2 - l13^2 with p1 = (l01,0,0):
    //   x3 = (l01^2 + l03^2 - l13^2) / (2*l01)
    // From |p3-p0|^2 - |p3-p2|^2 = l03^2 - l23^2 with p2 = (x2,y2,0):
    //   x3^2+y3^2+z3^2 - ((x3-x2)^2+(y3-y2)^2+z3^2) = l03^2 - l23^2
    //   2*x2*x3 + 2*y2*y3 - x2^2 - y2^2 = l03^2 - l23^2
    //   y3 = (l03^2 - l23^2 + x2^2 + y2^2 - 2*x2*x3) / (2*y2)
    let x3 = (l01.powi(2) + l03.powi(2) - l13.powi(2)) / (2.0 * l01);
    let y3 = if y2.abs() > 1e-12 {
        (l03.powi(2) - l23.powi(2) + x2.powi(2) + y2.powi(2) - 2.0 * x2 * x3) / (2.0 * y2)
    } else {
        0.0
    };
    let z3_sq = l03.powi(2) - x3.powi(2) - y3.powi(2);
    if z3_sq < -1e-9 {
        return None;
    }
    let z3 = z3_sq.max(0.0).sqrt();
    let p3 = Vector3::new(x3, y3, z3);

    Some([p0, p1, p2, p3])
}

/// Dihedral angle of tetrahedron `t` at edge `e`, computed from the local
/// embedding. The dihedral angle at edge (a,b) is the angle between the two
/// faces of the tetrahedron that share that edge, measured through the
/// tetrahedron's interior.
fn dihedral_angle_at_edge(lengths: &EdgeLengths, t: &Tetrahedron, e: &Edge) -> Option<f64> {
    let pts = embed_tetrahedron(lengths, t)?;
    let idx_of = |vid: VertexId| t.v.iter().position(|&x| x == vid).unwrap();
    let ia = idx_of(e.0);
    let ib = idx_of(e.1);
    // The other two vertices (opposite the hinge edge).
    let others: Vec<usize> = (0..4).filter(|&i| i != ia && i != ib).collect();
    let (ic, id) = (others[0], others[1]);

    let a = pts[ia];
    let b = pts[ib];
    let c = pts[ic];
    let d = pts[id];

    let ab = b - a;
    let ab_dir = ab.normalize();
    // Project c and d onto the plane perpendicular to edge ab, through a.
    let ac = c - a;
    let ad = d - a;
    let c_perp = ac - ab_dir * ac.dot(&ab_dir);
    let d_perp = ad - ab_dir * ad.dot(&ab_dir);
    if c_perp.norm() < 1e-12 || d_perp.norm() < 1e-12 {
        return None; // degenerate tetrahedron
    }
    let cos_theta = c_perp.dot(&d_perp) / (c_perp.norm() * d_perp.norm());
    Some(cos_theta.clamp(-1.0, 1.0).acos())
}

/// Deficit angle at a hinge (edge) `e`: 2*pi minus the sum of dihedral
/// angles of every tetrahedron incident to `e`. This is the discrete
/// curvature 2-form of Regge calculus, concentrated entirely on hinges
/// (everywhere else the piecewise-flat manifold is, by construction, flat).
pub fn deficit_angle(
    complex: &SimplicialComplex,
    lengths: &EdgeLengths,
    e: &Edge,
) -> Option<f64> {
    let tets = complex.edge_to_tets.get(e)?;
    if tets.len() < 3 {
        return None; // not a well-posed interior hinge under our stated convention
    }
    let mut sum = 0.0;
    for &ti in tets {
        let theta = dihedral_angle_at_edge(lengths, &complex.tetrahedra[ti], e)?;
        sum += theta;
    }
    Some(2.0 * std::f64::consts::PI - sum)
}

/// The Regge action: S = sum_hinges L_hinge * deficit(hinge), the exact
/// discretization of integral(R sqrt(g)) for a piecewise-flat 3-manifold.
/// Optionally include a cosmological term +2*Lambda*sum(tet volumes),
/// mirroring integral((R - 2*Lambda) sqrt(g)).
pub struct ReggeActionResult {
    pub curvature_term: f64,
    pub volume_term: f64,
    pub total: f64,
    pub n_hinges_used: usize,
    pub n_hinges_skipped_boundary: usize,
}

pub fn regge_action(
    complex: &SimplicialComplex,
    lengths: &EdgeLengths,
    lambda: f64,
) -> ReggeActionResult {
    let mut curvature_term = 0.0;
    let mut n_used = 0;
    let mut n_skipped = 0;
    for e in &complex.edges {
        match deficit_angle(complex, lengths, e) {
            Some(delta) => {
                let l = lengths.get(e.0, e.1);
                curvature_term += l * delta;
                n_used += 1;
            }
            None => n_skipped += 1,
        }
    }
    let mut volume_term = 0.0;
    for t in &complex.tetrahedra {
        volume_term += tetrahedron_volume(lengths, t);
    }
    let volume_term = 2.0 * lambda * volume_term;
    ReggeActionResult {
        curvature_term,
        volume_term,
        total: curvature_term + volume_term,
        n_hinges_used: n_used,
        n_hinges_skipped_boundary: n_skipped,
    }
}

/// Check that every tetrahedron's edge lengths currently satisfy the
/// tetrahedron inequality (Cayley-Menger positivity) -- i.e. that this
/// EdgeLengths assignment corresponds to a genuine piecewise-flat geometry.
pub fn all_tetrahedra_valid(complex: &SimplicialComplex, lengths: &EdgeLengths) -> bool {
    complex
        .tetrahedra
        .iter()
        .all(|t| is_valid_tetrahedron(lengths, t))
}
