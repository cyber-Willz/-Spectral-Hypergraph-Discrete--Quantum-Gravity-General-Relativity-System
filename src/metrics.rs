//! Canonical exact solutions of the Einstein field equations, as metric
//! callbacks compatible with `tensor_calculus::curvature_at`.
//!
//! Each metric here comes with an independently-known closed-form curvature
//! result (from standard GR textbooks, not derived by this crate) that the
//! numerical engine in `tensor_calculus.rs` is cross-checked against in the
//! test suite below -- the same "two independent derivations must agree"
//! pattern used throughout this crate (see `ihara_zeta.rs`, `regge_eom.rs`).
//!
//! Units and conventions: `c = G = 1`, signature `(-,+,+,+)`, matching
//! `tensor_calculus.rs`.
//!
//! What this module does NOT claim:
//!   - Not an exhaustive solution catalog: no Kerr (rotating), no
//!     Reissner-Nordstrom (charged), no interior/matched stellar solutions.
//!     Schwarzschild and FRW were chosen because they have the best-known,
//!     least-ambiguous textbook closed-form curvature invariants to check
//!     against, not because the list is complete.
//!   - The FRW scale factor `a(t)` is caller-supplied and only required to
//!     be twice-differentiable; no Friedmann-equation matter sector (no
//!     energy density/pressure) is implied or solved for here -- this is
//!     "compute curvature of a given a(t)", not "find a(t) given matter
//!     content".

use nalgebra::Matrix4;

use crate::tensor_calculus::Point4;

/// Schwarzschild metric in standard Schwarzschild coordinates
/// `x = [t, r, theta, phi]`, parameterized directly by the Schwarzschild
/// radius `r_s = 2M` (so `M = r_s/2` in these `G=c=1` units). Valid for
/// `r > r_s` (outside the horizon); the caller is responsible for evaluating
/// only there.
pub fn schwarzschild(r_s: f64) -> impl Fn(&Point4) -> Matrix4<f64> {
    move |x: &Point4| {
        let r = x[1];
        let theta = x[2];
        let f = 1.0 - r_s / r;
        Matrix4::new(
            -f, 0.0, 0.0, 0.0, //
            0.0, 1.0 / f, 0.0, 0.0, //
            0.0, 0.0, r * r, 0.0, //
            0.0, 0.0, 0.0, r * r * theta.sin().powi(2),
        )
    }
}

/// Exact Kretschmann scalar for Schwarzschild, `K = 48 M^2 / r^6 = 12 r_s^2
/// / r^6` (standard textbook result, e.g. Misner-Thorne-Wheeler). Used as
/// the independent cross-check for `tensor_calculus::kretschmann_scalar`.
pub fn schwarzschild_kretschmann_exact(r_s: f64, r: f64) -> f64 {
    12.0 * r_s * r_s / r.powi(6)
}

/// Flat/open/closed FRW metric in comoving coordinates
/// `x = [t, chi, theta, phi]`, `ds^2 = -dt^2 + a(t)^2 [dchi^2/(1-k*chi^2) +
/// chi^2 dOmega^2]`. `k` in `{-1, 0, +1}` (open/flat/closed); `a` is an
/// arbitrary caller-supplied (twice-differentiable) scale factor.
///
/// Valid for `1 - k*chi^2 > 0` (always true for `k <= 0`; for `k = +1`
/// requires `chi < 1`, the closed-universe coordinate range).
pub fn frw(k: f64, a: impl Fn(f64) -> f64 + 'static) -> impl Fn(&Point4) -> Matrix4<f64> {
    move |x: &Point4| {
        let t = x[0];
        let chi = x[1];
        let theta = x[2];
        let at = a(t);
        let at2 = at * at;
        Matrix4::new(
            -1.0, 0.0, 0.0, 0.0, //
            0.0, at2 / (1.0 - k * chi * chi), 0.0, 0.0, //
            0.0, 0.0, at2 * chi * chi, 0.0, //
            0.0, 0.0, 0.0, at2 * chi * chi * theta.sin().powi(2),
        )
    }
}

/// Exact FRW Ricci scalar, `R = 6[a''/a + (a'/a)^2 + k/a^2]` (standard
/// textbook result, e.g. Wald *General Relativity* or Weinberg), given the
/// scale factor's value and its first two derivatives at time `t`.
pub fn frw_ricci_scalar_exact(k: f64, a: f64, a_dot: f64, a_ddot: f64) -> f64 {
    6.0 * (a_ddot / a + (a_dot / a).powi(2) + k / (a * a))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor_calculus::curvature_at;

    /// Schwarzschild is a vacuum solution: R_{ab} = 0 everywhere outside the
    /// horizon (this is, in fact, the *defining* property used to originally
    /// derive it). Checked at an interior point away from the horizon/pole
    /// coordinate singularities, where the finite-difference stencil is well
    /// behaved.
    #[test]
    fn schwarzschild_is_ricci_flat() {
        let r_s = 1.0;
        let metric = schwarzschild(r_s);
        let x = [0.0, 10.0 * r_s, std::f64::consts::FRAC_PI_3, 0.7];
        let c = curvature_at(&metric, &x, 1e-4);
        for i in 0..4 {
            for j in 0..4 {
                assert!(
                    c.ricci[(i, j)].abs() < 1e-3,
                    "Schwarzschild R_{{{i}{j}}} = {} should be ~0 (vacuum)",
                    c.ricci[(i, j)]
                );
            }
        }
        assert!(c.ricci_scalar.abs() < 1e-3);
    }

    /// The Kretschmann scalar is the genuinely non-trivial cross-check:
    /// unlike Ricci, it doesn't vanish in vacuum, so agreement here actually
    /// exercises the Riemann tensor computation, not just "everything is
    /// numerically near zero."
    #[test]
    fn schwarzschild_kretschmann_matches_closed_form() {
        let r_s = 1.0;
        let metric = schwarzschild(r_s);
        for &r in &[5.0, 10.0, 25.0] {
            let x = [0.0, r, std::f64::consts::FRAC_PI_4, 0.3];
            let c = curvature_at(&metric, &x, 1e-4);
            let exact = schwarzschild_kretschmann_exact(r_s, r);
            let rel_err = (c.kretschmann - exact).abs() / exact;
            assert!(
                rel_err < 1e-2,
                "r={r}: numeric K={}, exact K={exact}, rel err={rel_err}",
                c.kretschmann
            );
        }
    }

    /// Matter-dominated flat FRW, a(t) = t^(2/3): a well-known exact
    /// Friedmann solution, chosen because a'/a and a''/a have simple closed
    /// forms to check the numeric engine against.
    #[test]
    fn frw_ricci_scalar_matches_closed_form_matter_dominated() {
        let a_fn = |t: f64| t.powf(2.0 / 3.0);
        let metric = frw(0.0, a_fn);
        let t = 2.5;
        let x = [t, 0.4, std::f64::consts::FRAC_PI_3, 0.2];
        let c = curvature_at(&metric, &x, 1e-4);

        let a = a_fn(t);
        let a_dot = (2.0 / 3.0) * t.powf(-1.0 / 3.0);
        let a_ddot = -(2.0 / 9.0) * t.powf(-4.0 / 3.0);
        let exact = frw_ricci_scalar_exact(0.0, a, a_dot, a_ddot);

        let rel_err = (c.ricci_scalar - exact).abs() / exact.abs();
        assert!(
            rel_err < 1e-2,
            "numeric R={}, exact R={exact}, rel err={rel_err}",
            c.ricci_scalar
        );
    }

    /// FRW is spatially homogeneous and isotropic by construction: the Ricci
    /// scalar at fixed cosmic time t must not depend on (chi, theta, phi).
    /// This is a structural check independent of the closed-form formula
    /// above -- it would catch a bug that broke homogeneity even if it
    /// somehow preserved the value at one point.
    #[test]
    fn frw_ricci_scalar_is_spatially_homogeneous() {
        let a_fn = |t: f64| t.powf(2.0 / 3.0);
        let metric = frw(0.0, a_fn);
        let t = 3.0;
        let points = [
            [t, 0.2, 0.5, 0.1],
            [t, 0.6, 1.2, 2.0],
            [t, 0.9, 2.4, -1.0],
        ];
        let values: Vec<f64> = points
            .iter()
            .map(|x| curvature_at(&metric, x, 1e-4).ricci_scalar)
            .collect();
        for w in values.windows(2) {
            let rel_diff = (w[0] - w[1]).abs() / w[0].abs().max(1e-12);
            assert!(
                rel_diff < 1e-2,
                "Ricci scalar should be spatially constant on FRW: {} vs {}",
                w[0],
                w[1]
            );
        }
    }
}
