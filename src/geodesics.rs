//! Geodesic integration: the equation of motion
//! `d^2 x^a / d\lambda^2 + Gamma^a_{bc} (dx^b/d\lambda)(dx^c/d\lambda) = 0`,
//! built entirely on the numerical Christoffel symbols from
//! `tensor_calculus.rs` -- i.e. this works for *any* metric, not just
//! Schwarzschild, exactly like the curvature engine itself.
//!
//! Verification strategy (same "independent check" pattern as the rest of
//! this crate):
//!   1. **Norm conservation.** `g_{ab} u^a u^b` is constant along any
//!      geodesic (a structural fact, true for every metric) -- checked for
//!      both a flat (Minkowski) and curved (Schwarzschild) trajectory as a
//!      basic integrator-correctness test.
//!   2. **Killing conserved quantities.** Schwarzschild has time-translation
//!      and axial Killing vectors, giving two *emergent* conserved
//!      quantities (`E = -g_{tt} u^t`, `L = g_{phiphi} u^phi`) that are not
//!      hard-coded into the integrator -- they fall out of the geodesic
//!      equation and the specific symmetry of this metric. Their near-
//!      constancy along an integrated trajectory is a non-trivial check on
//!      both the Christoffel computation and the integrator.
//!   3. **Light bending.** A textbook, independently-known weak-field
//!      result: a photon passing a mass at impact parameter `b >> r_s`
//!      deflects by `Delta\phi ~ 2 r_s / b` (`= 4GM/(c^2 b)` restoring
//!      units). Integrating a genuine null geodesic through closest approach
//!      and measuring the deflection is the closest thing in this crate to
//!      an end-to-end continuum-GR prediction test.
//!
//! What this module does NOT claim:
//!   - Fixed-step RK4 only -- no adaptive step control or symplectic
//!     integrator, so trajectories very close to the photon sphere/horizon
//!     (where curvature changes fast) will need a smaller step than what's
//!     used for the weak-field test below, and this module doesn't do that
//!     automatically.
//!   - No general turning-point/horizon detection beyond the specific
//!     "radius crosses back above its start" logic used for the light
//!     bending test, which is fit to that single scenario, not general
//!     enough to be a solver for arbitrary geodesic problems.

use nalgebra::Matrix4;

use crate::tensor_calculus::{christoffel, Point4};

/// A point in phase space: position `x^a` plus velocity `u^a = dx^a/d\lambda`.
#[derive(Clone, Copy, Debug)]
pub struct GeodesicState {
    pub x: Point4,
    pub u: Point4,
}

/// Right-hand side of the geodesic equation: returns `(dx/d\lambda,
/// du/d\lambda) = (u, -Gamma^a_{bc} u^b u^c)`.
fn rhs(
    metric: &dyn Fn(&Point4) -> Matrix4<f64>,
    state: &GeodesicState,
    h: f64,
) -> (Point4, Point4) {
    let gamma = christoffel(metric, &state.x, h);
    let u = state.u;
    let mut du = [0.0_f64; 4];
    for a in 0..4 {
        let mut sum = 0.0;
        for b in 0..4 {
            for c in 0..4 {
                sum += gamma[a][b][c] * u[b] * u[c];
            }
        }
        du[a] = -sum;
    }
    (u, du)
}

fn add_scaled(a: &Point4, b: &Point4, s: f64) -> Point4 {
    [a[0] + s * b[0], a[1] + s * b[1], a[2] + s * b[2], a[3] + s * b[3]]
}

/// One classical RK4 step of size `dlambda`.
pub fn rk4_step(
    metric: &dyn Fn(&Point4) -> Matrix4<f64>,
    state: &GeodesicState,
    dlambda: f64,
    h: f64,
) -> GeodesicState {
    let (k1x, k1u) = rhs(metric, state, h);

    let s2 = GeodesicState {
        x: add_scaled(&state.x, &k1x, dlambda / 2.0),
        u: add_scaled(&state.u, &k1u, dlambda / 2.0),
    };
    let (k2x, k2u) = rhs(metric, &s2, h);

    let s3 = GeodesicState {
        x: add_scaled(&state.x, &k2x, dlambda / 2.0),
        u: add_scaled(&state.u, &k2u, dlambda / 2.0),
    };
    let (k3x, k3u) = rhs(metric, &s3, h);

    let s4 = GeodesicState {
        x: add_scaled(&state.x, &k3x, dlambda),
        u: add_scaled(&state.u, &k3u, dlambda),
    };
    let (k4x, k4u) = rhs(metric, &s4, h);

    let mut x = state.x;
    let mut u = state.u;
    for i in 0..4 {
        x[i] += (dlambda / 6.0) * (k1x[i] + 2.0 * k2x[i] + 2.0 * k3x[i] + k4x[i]);
        u[i] += (dlambda / 6.0) * (k1u[i] + 2.0 * k2u[i] + 2.0 * k3u[i] + k4u[i]);
    }
    GeodesicState { x, u }
}

/// Integrate `steps` RK4 steps of size `dlambda`, returning the full
/// trajectory (including the initial state).
pub fn integrate(
    metric: &dyn Fn(&Point4) -> Matrix4<f64>,
    initial: GeodesicState,
    dlambda: f64,
    steps: usize,
    h: f64,
) -> Vec<GeodesicState> {
    let mut traj = Vec::with_capacity(steps + 1);
    traj.push(initial);
    let mut state = initial;
    for _ in 0..steps {
        state = rk4_step(metric, &state, dlambda, h);
        traj.push(state);
    }
    traj
}

/// `g_{ab} u^a u^b` at a state -- constant along any geodesic (0 for null,
/// -1 for a unit-normalized timelike geodesic, +1 for unit spacelike).
pub fn norm(metric: &dyn Fn(&Point4) -> Matrix4<f64>, state: &GeodesicState) -> f64 {
    let g = metric(&state.x);
    let mut n = 0.0;
    for a in 0..4 {
        for b in 0..4 {
            n += g[(a, b)] * state.u[a] * state.u[b];
        }
    }
    n
}

/// Set up a null (photon) geodesic in equatorial Schwarzschild (`theta =
/// pi/2`) starting at large radius `r0`, incoming with impact parameter `b`
/// (energy `E=1` at infinity, angular momentum `L=b`), and integrate it
/// through closest approach back out to `r0`. Returns the deflection angle:
/// the total `phi` swept, minus the *flat-space* sweep between the same two
/// finite-radius points, `pi - 2 asin(b/r0)` (exactly `pi` only in the
/// `r0 -> infinity` limit -- at finite `r0` a straight line from `r=r0` to
/// `r=r0` past a perpendicular offset `b` already subtends less than `pi`,
/// purely as flat-space geometry, and that has to be subtracted out before
/// what's left can be called "the GR effect"). The weak-field prediction
/// for the *remaining* (genuinely gravitational) piece is `~= 2 r_s / b`
/// for `b >> r_s`.
pub fn schwarzschild_light_deflection(r_s: f64, b: f64, r0: f64, dlambda: f64) -> f64 {
    use crate::metrics::schwarzschild;
    let metric = schwarzschild(r_s);

    let f0 = 1.0 - r_s / r0;
    let e = 1.0; // energy at infinity
    let l = b; // angular momentum (b = L/E, E=1)
    let u_t = e / f0;
    let u_phi = l / (r0 * r0);
    // Null condition: -f (u^t)^2 + (u^r)^2/f + r^2 (u^phi)^2 = 0, incoming (u^r < 0).
    let u_r_sq = f0 * (f0 * u_t * u_t - r0 * r0 * u_phi * u_phi);
    let u_r = -(u_r_sq.max(0.0)).sqrt();

    let mut state = GeodesicState {
        x: [0.0, r0, std::f64::consts::FRAC_PI_2, 0.0],
        u: [u_t, u_r, 0.0, u_phi],
    };

    let h = 1e-4;
    let mut went_inward = false;
    let max_steps = 50_000_000usize;
    let mut steps_taken = 0usize;
    loop {
        state = rk4_step(&metric, &state, dlambda, h);
        steps_taken += 1;
        if state.x[1] < r0 * 0.999 {
            went_inward = true;
        }
        if went_inward && state.x[1] > r0 * 0.999 {
            break;
        }
        if steps_taken >= max_steps {
            break; // give up rather than loop forever; caller sees a stale phi
        }
    }
    let flat_baseline = std::f64::consts::PI - 2.0 * (b / r0).asin();
    state.x[3] - flat_baseline
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::Matrix4;

    fn minkowski(_x: &Point4) -> Matrix4<f64> {
        Matrix4::new(
            -1.0, 0.0, 0.0, 0.0, //
            0.0, 1.0, 0.0, 0.0, //
            0.0, 0.0, 1.0, 0.0, //
            0.0, 0.0, 0.0, 1.0,
        )
    }

    /// In flat spacetime a geodesic is a straight line at constant velocity;
    /// this is the simplest possible correctness check on the integrator
    /// before trusting it on anything curved.
    #[test]
    fn minkowski_geodesic_is_a_straight_line() {
        let initial = GeodesicState {
            x: [0.0, 0.0, 0.0, 0.0],
            u: [1.0, 0.3, 0.2, 0.1],
        };
        let traj = integrate(&minkowski, initial, 0.1, 100, 1e-4);
        let last = traj.last().unwrap();
        for i in 0..4 {
            let expected = initial.x[i] + initial.u[i] * (100.0 * 0.1);
            assert!(
                (last.x[i] - expected).abs() < 1e-6,
                "component {i}: got {}, expected {expected}",
                last.x[i]
            );
        }
    }

    /// Norm conservation on a curved (Schwarzschild) timelike geodesic: a
    /// circular-orbit-adjacent trajectory, checked over many steps.
    #[test]
    fn schwarzschild_timelike_geodesic_conserves_norm() {
        use crate::metrics::schwarzschild;
        let r_s = 1.0;
        let metric = schwarzschild(r_s);
        let r0 = 10.0 * r_s;
        // Circular-orbit angular velocity for Schwarzschild: (d\phi/dt)^2 = M/r^3.
        let m = r_s / 2.0;
        let omega = (m / r0.powi(3)).sqrt();
        let f0 = 1.0 - r_s / r0;
        // Normalize u^t so that the norm is -1 (proper-time parametrization):
        // -f (u^t)^2 + r^2 (u^phi)^2 = -1, u^phi = omega * u^t.
        let u_t = (1.0 / (f0 - r0 * r0 * omega * omega)).sqrt();
        let u_phi = omega * u_t;
        let initial = GeodesicState {
            x: [0.0, r0, std::f64::consts::FRAC_PI_2, 0.0],
            u: [u_t, 0.0, 0.0, u_phi],
        };
        let n0 = norm(&metric, &initial);
        assert!((n0 + 1.0).abs() < 1e-3, "initial norm should be ~-1, got {n0}");

        let traj = integrate(&metric, initial, 0.01, 2000, 1e-4);
        let last = traj.last().unwrap();
        let n_final = norm(&metric, last);
        assert!(
            (n_final - n0).abs() < 1e-2,
            "norm drifted: {n0} -> {n_final}"
        );
    }

    /// Killing-vector conserved quantities along a Schwarzschild geodesic:
    /// E = -g_tt u^t and L = g_phiphi u^phi should each stay close to their
    /// initial values, even though nothing in the integrator enforces this
    /// directly -- it's an emergent consequence of the metric's symmetry.
    #[test]
    fn schwarzschild_geodesic_conserves_energy_and_angular_momentum() {
        use crate::metrics::schwarzschild;
        let r_s = 1.0;
        let metric = schwarzschild(r_s);
        let r0 = 8.0 * r_s;
        let f0 = 1.0 - r_s / r0;
        // A moderately eccentric-looking initial condition (nonzero radial
        // velocity) rather than a pure circular orbit, so the check isn't
        // trivially easy.
        let u_t = 1.05 / f0;
        let u_phi = 0.6 / (r0 * r0);
        let u_r_sq = f0 * (f0 * u_t * u_t - r0 * r0 * u_phi * u_phi - 1.0);
        assert!(u_r_sq > 0.0, "test setup should give a valid timelike geodesic");
        let u_r = u_r_sq.sqrt();
        let initial = GeodesicState {
            x: [0.0, r0, std::f64::consts::FRAC_PI_2, 0.0],
            u: [u_t, u_r, 0.0, u_phi],
        };

        let e0 = f0 * u_t; // -g_tt u^t = f * u^t
        let l0 = r0 * r0 * u_phi; // g_phiphi u^phi

        let traj = integrate(&metric, initial, 0.005, 3000, 1e-4);
        for state in traj.iter().step_by(500) {
            let g = metric(&state.x);
            let f = -g[(0, 0)];
            let e = f * state.u[0];
            let l = g[(3, 3)] * state.u[3];
            assert!((e - e0).abs() / e0 < 1e-2, "E drifted: {e0} -> {e}");
            assert!((l - l0).abs() / l0 < 1e-2, "L drifted: {l0} -> {l}");
        }
    }

    /// Weak-field light bending: b >> r_s, so the exact GR deflection
    /// `Delta\phi` should be close to the standard weak-field prediction
    /// `2 r_s / b`. This is the closest thing in this crate to an
    /// end-to-end "integrate a real geodesic, compare to a textbook GR
    /// number" test.
    #[test]
    fn light_bending_matches_weak_field_prediction() {
        let r_s = 1.0;
        let b = 50.0 * r_s;
        let r0 = 200.0 * r_s;
        let deflection = schwarzschild_light_deflection(r_s, b, r0, 0.005);
        let predicted = 2.0 * r_s / b;
        let rel_err = (deflection - predicted).abs() / predicted;
        assert!(
            rel_err < 0.1,
            "numeric deflection {deflection}, weak-field prediction {predicted}, rel err {rel_err}"
        );
    }

    /// Same physical setup, much larger r0/b so b/r0 -> 0 and the flat-space
    /// baseline correction above shrinks to noise: confirms the finite-r0
    /// baseline formula itself (not just the b=50,r0=200 case) by agreeing
    /// with the classic textbook asymptotic statement "Delta\phi = 2 r_s/b"
    /// under the regime where that statement is literally, not just
    /// approximately, the right comparison.
    #[test]
    fn light_bending_deflection_is_stable_across_r0_choices() {
        let r_s = 1.0;
        let b = 50.0 * r_s;
        let predicted = 2.0 * r_s / b;
        for &r0 in &[200.0, 500.0, 1000.0] {
            let deflection = schwarzschild_light_deflection(r_s, b, r0, 0.005);
            let rel_err = (deflection - predicted).abs() / predicted;
            assert!(
                rel_err < 0.1,
                "r0={r0}: numeric deflection {deflection}, predicted {predicted}, rel err {rel_err}"
            );
        }
    }
}
