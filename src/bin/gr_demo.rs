//! End-to-end demo of the continuum GR modules (`tensor_calculus.rs`,
//! `metrics.rs`, `geodesics.rs`), in the same spirit as `main.rs` and
//! `regge_demo.rs`: print real numbers computed live, not narrated claims.

use nalgebra::Matrix4;
use spectral_dqg::geodesics::{integrate, norm, schwarzschild_light_deflection, GeodesicState};
use spectral_dqg::metrics::{
    frw, frw_ricci_scalar_exact, schwarzschild, schwarzschild_kretschmann_exact,
};
use spectral_dqg::tensor_calculus::{curvature_at, Point4, DEFAULT_H};

fn minkowski(_x: &Point4) -> Matrix4<f64> {
    Matrix4::new(
        -1.0, 0.0, 0.0, 0.0, //
        0.0, 1.0, 0.0, 0.0, //
        0.0, 0.0, 1.0, 0.0, //
        0.0, 0.0, 0.0, 1.0,
    )
}

fn main() {
    println!("================================================================");
    println!(" Continuum GR: tensor calculus engine, exact solutions, geodesics");
    println!("================================================================\n");

    // ---- Step 1: Minkowski sanity check ----------------------------------
    println!("---- Step 1: Minkowski (flat) -- every curvature tensor should vanish ----\n");
    let x = [0.3, 1.7, -0.5, 2.2];
    let c = minkowski_curvature_report(&x);
    println!(
        "max |R^a_bcd| = {:.3e}, Ricci scalar = {:.3e}, Kretschmann = {:.3e} (all expect ~0)",
        c.0, c.1, c.2
    );

    // ---- Step 2: Schwarzschild vacuum + Kretschmann -----------------------
    println!("\n---- Step 2: Schwarzschild -- vacuum check (R_ab=0) + Kretschmann scalar ----\n");
    let r_s = 1.0;
    let metric = schwarzschild(r_s);
    println!("{:>8} {:>14} {:>16} {:>16} {:>10}", "r/r_s", "max|R_ab|", "K numeric", "K exact 12rs^2/r^6", "rel err");
    for &r in &[3.0, 5.0, 10.0, 25.0, 50.0] {
        let xx = [0.0, r * r_s, std::f64::consts::FRAC_PI_3, 0.4];
        let curv = curvature_at(&metric, &xx, DEFAULT_H);
        let max_ricci = curv.ricci.iter().cloned().fold(0.0_f64, |a, b| a.max(b.abs()));
        let exact = schwarzschild_kretschmann_exact(r_s, r * r_s);
        let rel_err = (curv.kretschmann - exact).abs() / exact;
        println!(
            "{:>8.1} {:>14.3e} {:>16.6e} {:>16.6e} {:>9.3}%",
            r, max_ricci, curv.kretschmann, exact, rel_err * 100.0
        );
    }

    // ---- Step 3: FRW Ricci scalar vs closed-form Friedmann formula --------
    println!("\n---- Step 3: FRW (matter-dominated, a(t)=t^(2/3)) -- Ricci scalar vs closed form ----\n");
    let a_fn = |t: f64| t.powf(2.0 / 3.0);
    let frw_metric = frw(0.0, a_fn);
    println!("{:>6} {:>14} {:>14} {:>10}", "t", "R numeric", "R exact", "rel err");
    for &t in &[1.0, 2.5, 5.0, 10.0] {
        let xx = [t, 0.4, std::f64::consts::FRAC_PI_3, 0.2];
        let curv = curvature_at(&frw_metric, &xx, DEFAULT_H);
        let a = a_fn(t);
        let a_dot = (2.0 / 3.0) * t.powf(-1.0 / 3.0);
        let a_ddot = -(2.0 / 9.0) * t.powf(-4.0 / 3.0);
        let exact = frw_ricci_scalar_exact(0.0, a, a_dot, a_ddot);
        let rel_err = (curv.ricci_scalar - exact).abs() / exact.abs();
        println!("{:>6.1} {:>14.6} {:>14.6} {:>9.3}%", t, curv.ricci_scalar, exact, rel_err * 100.0);
    }
    println!(
        "\n(Homogeneity check: evaluating at three different (chi,theta,phi) points\n\
         at t=5.0 should give the same Ricci scalar -- a structural consequence of\n\
         FRW symmetry, independent of the closed-form comparison above.)"
    );
    for chi_theta_phi in &[[0.2, 0.5, 0.1], [0.6, 1.2, 2.0], [0.9, 2.4, -1.0]] {
        let xx = [5.0, chi_theta_phi[0], chi_theta_phi[1], chi_theta_phi[2]];
        let curv = curvature_at(&frw_metric, &xx, DEFAULT_H);
        println!("  chi={:.1} theta={:.1} phi={:.1} -> R = {:.6}", chi_theta_phi[0], chi_theta_phi[1], chi_theta_phi[2], curv.ricci_scalar);
    }

    // ---- Step 4: geodesic integration --------------------------------------
    println!("\n---- Step 4: geodesic integration -- norm & Killing-charge conservation ----\n");
    let r0 = 8.0 * r_s;
    let f0 = 1.0 - r_s / r0;
    let u_t = 1.05 / f0;
    let u_phi = 0.6 / (r0 * r0);
    let u_r_sq = f0 * (f0 * u_t * u_t - r0 * r0 * u_phi * u_phi - 1.0);
    let u_r = u_r_sq.sqrt();
    let initial = GeodesicState {
        x: [0.0, r0, std::f64::consts::FRAC_PI_2, 0.0],
        u: [u_t, u_r, 0.0, u_phi],
    };
    let n0 = norm(&metric, &initial);
    let e0 = f0 * u_t;
    let l0 = r0 * r0 * u_phi;
    println!("initial: norm={n0:.6} (expect -1, timelike), E={e0:.6}, L={l0:.6}");
    let traj = integrate(&metric, initial, 0.005, 3000, 1e-4);
    println!("{:>6} {:>10} {:>12} {:>12} {:>12}", "step", "r", "norm", "E", "L");
    for (i, state) in traj.iter().enumerate().step_by(500) {
        let g = metric(&state.x);
        let n = norm(&metric, state);
        let f = -g[(0, 0)];
        let e = f * state.u[0];
        let l = g[(3, 3)] * state.u[3];
        println!("{:>6} {:>10.4} {:>12.8} {:>12.8} {:>12.8}", i, state.x[1], n, e, l);
    }

    // ---- Step 5: light bending ----------------------------------------------
    println!("\n---- Step 5: light bending -- full null-geodesic integration vs weak-field formula ----\n");
    println!("{:>8} {:>10} {:>14} {:>14} {:>10}", "r0/r_s", "b/r_s", "deflection", "predicted 2rs/b", "rel err");
    let b = 50.0 * r_s;
    let predicted = 2.0 * r_s / b;
    for &r0_mult in &[200.0, 500.0, 1000.0] {
        let r0 = r0_mult * r_s;
        let deflection = schwarzschild_light_deflection(r_s, b, r0, 0.005);
        let rel_err = (deflection - predicted).abs() / predicted;
        println!(
            "{:>8.0} {:>10.1} {:>14.6} {:>14.6} {:>9.2}%",
            r0_mult, b / r_s, deflection, predicted, rel_err * 100.0
        );
    }
    println!(
        "\n(Deflection is computed as the total swept angle minus the *finite-r0*\n\
         flat-space baseline pi - 2*asin(b/r0), not bare pi -- see README for the\n\
         bug this distinction fixed. Stable across three different r0 choices.)"
    );

    println!("\n================================================================");
    println!(" Done -- all numbers above computed live by this run, not cached.");
    println!("================================================================");
}

fn minkowski_curvature_report(x: &Point4) -> (f64, f64, f64) {
    let curv = curvature_at(&minkowski, x, DEFAULT_H);
    let mut max_riemann = 0.0_f64;
    for a in 0..4 {
        for b in 0..4 {
            for cc in 0..4 {
                for d in 0..4 {
                    max_riemann = max_riemann.max(curv.riemann[a][b][cc][d].abs());
                }
            }
        }
    }
    (max_riemann, curv.ricci_scalar.abs(), curv.kretschmann.abs())
}
