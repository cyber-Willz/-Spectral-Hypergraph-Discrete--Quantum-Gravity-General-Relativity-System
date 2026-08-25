//! End-to-end demo of the Regge calculus / quantum gravity module, in the
//! same spirit as `main.rs`'s existing demo: print real numbers computed
//! live, not narrated claims.

use spectral_dqg::regge::*;
use spectral_dqg::regge_pi::*;
use std::collections::HashMap;

fn cube_vertices() -> Vec<[f64; 3]> {
    (0u8..8)
        .map(|i| [((i >> 2) & 1) as f64, ((i >> 1) & 1) as f64, (i & 1) as f64])
        .collect()
}
fn dist(a: [f64; 3], b: [f64; 3]) -> f64 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
}
fn lengths_from_coords(c: &SimplicialComplex, coords: &[[f64; 3]]) -> EdgeLengths {
    let mut l = HashMap::new();
    for &e in &c.edges {
        l.insert(e, dist(coords[e.0], coords[e.1]));
    }
    EdgeLengths { lengths: l }
}

fn main() {
    println!("================================================================");
    println!(" Regge calculus: action, path integral, triangulation-independence");
    println!("================================================================\n");

    println!("---- Step 1: single regular tetrahedron, closed-form check ----\n");
    let l = 2.3;
    let tet = Tetrahedron { v: [0, 1, 2, 3] };
    let mut lengths = HashMap::new();
    for e in tet.edges() {
        lengths.insert(e, l);
    }
    let lengths = EdgeLengths { lengths };
    let v = tetrahedron_volume(&lengths, &tet);
    let expected_v = l.powi(3) / (6.0 * 2f64.sqrt());
    println!(
        "edge length {l}: computed volume = {v:.6}, closed-form l^3/(6*sqrt(2)) = {expected_v:.6}, |diff| = {:.2e}",
        (v - expected_v).abs()
    );

    println!("\n---- Step 2: flat cube, 6-tet decomposition (diagonal 0-7) ----\n");
    let coords = cube_vertices();
    let tets_a = vec![
        [0, 1, 3, 7], [0, 1, 5, 7], [0, 2, 3, 7],
        [0, 2, 6, 7], [0, 4, 5, 7], [0, 4, 6, 7],
    ];
    let complex_a = SimplicialComplex::from_tetrahedra(tets_a);
    let lengths_a = lengths_from_coords(&complex_a, &coords);
    let hinge_a = (0usize, 7usize);
    let delta_a = deficit_angle(&complex_a, &lengths_a, &hinge_a).unwrap();
    let action_a = regge_action(&complex_a, &lengths_a, 0.0);
    println!(
        "hinge (0,7): multiplicity={}, deficit angle={:.3e} rad (expect 0 exactly for flat space)",
        complex_a.hinge_multiplicity(&hinge_a), delta_a
    );
    println!("total Regge action S = {:.3e} (expect 0)", action_a.total);

    println!("\n---- Step 3: SAME flat cube, independent decomposition (diagonal 1-6) ----\n");
    let tets_b = vec![
        [1, 5, 7, 6], [1, 5, 4, 6], [1, 3, 7, 6],
        [1, 3, 2, 6], [1, 0, 4, 6], [1, 0, 2, 6],
    ];
    let complex_b = SimplicialComplex::from_tetrahedra(tets_b);
    let lengths_b = lengths_from_coords(&complex_b, &coords);
    let hinge_b = (1usize, 6usize);
    let delta_b = deficit_angle(&complex_b, &lengths_b, &hinge_b).unwrap();
    let action_b = regge_action(&complex_b, &lengths_b, 0.0);
    println!(
        "hinge (1,6): multiplicity={}, deficit angle={:.3e} rad",
        complex_b.hinge_multiplicity(&hinge_b), delta_b
    );
    println!("total Regge action S = {:.3e}", action_b.total);
    println!(
        "\n(Two totally different simplicial complexes describing the SAME flat cube\n\
         agree to {:.1e} -- the honest, narrowly-scoped discrete residue of\n\
         diffeomorphism invariance: physics of a flat region doesn't depend on\n\
         which triangulation describes it. This is NOT full continuum\n\
         diffeomorphism invariance for curved configurations, which Regge\n\
         calculus is known to only approximately recover in the continuum limit.)",
        (delta_a - delta_b).abs()
    );

    println!("\n---- Step 4: perturb the hinge -> curvature appears ----\n");
    let mut lengths_c = lengths_a.clone();
    let orig = *lengths_c.lengths.get(&hinge_a).unwrap();
    lengths_c.lengths.insert(hinge_a, orig * 1.15);
    let delta_c = deficit_angle(&complex_a, &lengths_c, &hinge_a).unwrap();
    let action_c = regge_action(&complex_a, &lengths_c, 0.0);
    println!(
        "stretch main diagonal by 15%: deficit angle = {delta_c:.5} rad, S = {:.5}",
        action_c.total
    );

    println!("\n---- Step 5: Euclidean path integral, Metropolis over edge lengths ----\n");
    println!("Z = integral D[length] exp(-kappa * S_Regge[length]), fixed connectivity (complex_a)\n");
    println!("{:>7} {:>14} {:>14} {:>10} {:>8}", "kappa", "<S> (signed)", "stderr", "accept%", "n");
    for &kappa in &[0.2, 1.0, 5.0] {
        let initial = lengths_from_coords(&complex_a, &coords);
        let cfg = McConfig { kappa, lambda: 0.0, step_size: 0.05, n_sweeps: 800, seed: 11 };
        let r = run_path_integral(&complex_a, initial, &cfg);
        let burn = 100;
        let mean: f64 = r.action_trace[burn..].iter().sum::<f64>() / (r.action_trace.len() - burn) as f64;
        println!(
            "{kappa:>7.2} {mean:>14.5} {:>14.5} {:>9.1}% {:>8}",
            r.stderr_action, r.acceptance_rate * 100.0, r.n_samples
        );
    }
    println!(
        "\n(<S> runs MORE negative as kappa grows, not toward 0 -- this is the\n\
         discrete conformal-factor problem (Gibbons-Hawking-Perry 1978): the\n\
         Euclidean curvature term sum(L*deficit) is unbounded below, so naive\n\
         real-weight exp(-kappa*S) sampling is pulled toward large negative-\n\
         curvature configurations, capped only by the hard tetrahedron-\n\
         validity constraint -- reproduced correctly here, not papered over.)"
    );
}
