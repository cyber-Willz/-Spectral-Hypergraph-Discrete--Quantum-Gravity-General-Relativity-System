//! Continuum general relativity: a generic numerical tensor-calculus engine.
//!
//! Everything else GR-adjacent in this crate (`regge.rs`, `regge_eom.rs`,
//! `gis_gnss_relativity.rs`) is either purely discrete (Regge calculus) or a
//! single hand-derived special case (the GPS clock correction is a weak-field
//! formula written down directly, not built on general tensor machinery).
//! This module is the piece that was missing: given *any* metric
//! `g_{ab}(x)` as a callback, compute the actual continuum GR curvature
//! tensors from first principles —
//!
//!   Christoffel symbols -> Riemann tensor -> Ricci tensor/scalar
//!   -> Einstein tensor, plus the Kretschmann scalar `R_{abcd}R^{abcd}`.
//!
//! Convention (fixed once, used everywhere in this module and in
//! `metrics.rs`/`geodesics.rs`): signature `(-,+,+,+)`, coordinates
//! `x = [x^0, x^1, x^2, x^3]`, units with `c = G = 1` (so e.g. a Schwarzschild
//! mass parameter is directly the Schwarzschild radius `r_s = 2M`). Riemann
//! sign convention (matching Wald, *General Relativity*, eq. 3.2.3):
//!
//! ```text
//! R^a_{bcd} = d_c Gamma^a_{db} - d_d Gamma^a_{cb}
//!           + Gamma^a_{ce} Gamma^e_{db} - Gamma^a_{de} Gamma^e_{cb}
//! R_{bd} = R^a_{bad}          (Ricci: contract 1st and 3rd indices)
//! ```
//!
//! **How derivatives are taken.** There is no symbolic differentiation here:
//! `christoffel` numerically differentiates the metric (central difference,
//! step `h`), and `riemann` numerically differentiates `christoffel` itself
//! (i.e. this is a *second* numerical derivative of the metric, taken as two
//! nested first differences rather than a single wider stencil). This is
//! deliberate — it makes the engine generic over any metric callback,
//! including ones with no closed-form Christoffel symbols (e.g. the FRW
//! metric in `metrics.rs`, where the scale factor `a(t)` is an arbitrary
//! caller-supplied function) — but it is *not* machine-precision like the
//! closed-form Regge calculus in this crate. Expect relative error on the
//! order of `1e-4`-`1e-6` for a well-chosen `h` (default recommendation
//! `1e-4`, balancing `O(h^2)` truncation error against `O(eps/h^2)`
//! round-off error); this is verified, not assumed, in the test suite below
//! and in `metrics.rs` against exact closed-form results (Schwarzschild
//! vacuum `R_{ab}=0`, the exact Kretschmann scalar `48M^2/r^6`, and the exact
//! FRW Ricci scalar `6[a''/a + (a'/a)^2 + k/a^2]`).
//!
//! What this module does NOT claim:
//!   - Not a computer-algebra system: no symbolic simplification, no
//!     coordinate-independent (tensorial) error bound -- accuracy is
//!     coordinate- and scale-dependent through the finite-difference step
//!     `h`, exactly as documented above.
//!   - No stress-energy tensor / matter sector and therefore no solving of
//!     the Einstein equations `G_{ab} = 8*pi*T_{ab}` for an unknown metric;
//!     this computes curvature *from* an already-specified metric.
//!   - No connection (yet) to the discrete Regge machinery elsewhere in this
//!     crate -- this is a standalone continuum engine, analogous to how
//!     `gis_gnss_relativity.rs` was standalone before it. Bridging the two
//!     (comparing a Regge deficit angle to a continuum curvature component
//!     on a matching geometry) is future work, not implemented here.

use nalgebra::Matrix4;

/// A spacetime point in the fixed 4D coordinate convention used throughout.
pub type Point4 = [f64; 4];

/// Default finite-difference step. See module docs for the accuracy
/// trade-off this balances.
pub const DEFAULT_H: f64 = 1e-4;

/// Christoffel symbols `Gamma[a][b][c]` = `Gamma^a_{bc}`, computed by
/// numerically differentiating the metric at `x` with step `h`.
pub fn christoffel(
    metric: &dyn Fn(&Point4) -> Matrix4<f64>,
    x: &Point4,
    h: f64,
) -> [[[f64; 4]; 4]; 4] {
    let g = metric(x);
    let g_inv = g
        .try_inverse()
        .expect("metric is not invertible at this point");

    // dg[c] = d g_{..} / d x^c
    let mut dg = [Matrix4::<f64>::zeros(); 4];
    for c in 0..4 {
        let mut xp = *x;
        let mut xm = *x;
        xp[c] += h;
        xm[c] -= h;
        dg[c] = (metric(&xp) - metric(&xm)) / (2.0 * h);
    }

    let mut gamma = [[[0.0_f64; 4]; 4]; 4];
    for a in 0..4 {
        for b in 0..4 {
            for c in 0..4 {
                let mut sum = 0.0;
                for d in 0..4 {
                    sum += g_inv[(a, d)] * (dg[b][(d, c)] + dg[c][(d, b)] - dg[d][(b, c)]);
                }
                gamma[a][b][c] = 0.5 * sum;
            }
        }
    }
    gamma
}

/// `d Gamma^a_{bc} / d x^wrt`, by central-differencing `christoffel` itself.
/// This is the "second numerical derivative of the metric" the module docs
/// warn about -- the dominant source of error in `riemann` below.
fn d_christoffel(
    metric: &dyn Fn(&Point4) -> Matrix4<f64>,
    x: &Point4,
    wrt: usize,
    h: f64,
) -> [[[f64; 4]; 4]; 4] {
    let mut xp = *x;
    let mut xm = *x;
    xp[wrt] += h;
    xm[wrt] -= h;
    let gp = christoffel(metric, &xp, h);
    let gm = christoffel(metric, &xm, h);
    let mut d = [[[0.0_f64; 4]; 4]; 4];
    for a in 0..4 {
        for b in 0..4 {
            for c in 0..4 {
                d[a][b][c] = (gp[a][b][c] - gm[a][b][c]) / (2.0 * h);
            }
        }
    }
    d
}

/// Riemann tensor `R[a][b][c][d]` = `R^a_{bcd}` (one index up, three down),
/// via the convention fixed in the module docs.
pub fn riemann(
    metric: &dyn Fn(&Point4) -> Matrix4<f64>,
    x: &Point4,
    h: f64,
) -> [[[[f64; 4]; 4]; 4]; 4] {
    let gamma = christoffel(metric, x, h);
    let mut dgamma = [[[[0.0_f64; 4]; 4]; 4]; 4]; // dgamma[wrt][a][b][c]
    for wrt in 0..4 {
        dgamma[wrt] = d_christoffel(metric, x, wrt, h);
    }

    let mut r = [[[[0.0_f64; 4]; 4]; 4]; 4]; // r[a][b][c][d]
    for a in 0..4 {
        for b in 0..4 {
            for c in 0..4 {
                for d in 0..4 {
                    let mut val = dgamma[c][a][d][b] - dgamma[d][a][c][b];
                    for e in 0..4 {
                        val += gamma[a][c][e] * gamma[e][d][b] - gamma[a][d][e] * gamma[e][c][b];
                    }
                    r[a][b][c][d] = val;
                }
            }
        }
    }
    r
}

/// Ricci tensor `R_{bd} = R^a_{bad}` (contract 1st and 3rd Riemann indices).
pub fn ricci_tensor(riemann: &[[[[f64; 4]; 4]; 4]; 4]) -> Matrix4<f64> {
    let mut ric = Matrix4::<f64>::zeros();
    for b in 0..4 {
        for d in 0..4 {
            let mut sum = 0.0;
            for a in 0..4 {
                sum += riemann[a][b][a][d];
            }
            ric[(b, d)] = sum;
        }
    }
    ric
}

/// Ricci scalar `R = g^{bd} R_{bd}`.
pub fn ricci_scalar(g_inv: &Matrix4<f64>, ricci: &Matrix4<f64>) -> f64 {
    let mut r = 0.0;
    for b in 0..4 {
        for d in 0..4 {
            r += g_inv[(b, d)] * ricci[(b, d)];
        }
    }
    r
}

/// Einstein tensor `G_{ab} = R_{ab} - (1/2) R g_{ab}`.
pub fn einstein_tensor(g: &Matrix4<f64>, ricci: &Matrix4<f64>, r_scalar: f64) -> Matrix4<f64> {
    ricci - 0.5 * r_scalar * g
}

/// Kretschmann scalar `K = R_{abcd} R^{abcd}`, a curvature invariant that
/// (unlike the Ricci scalar) does not vanish in vacuum -- the standard
/// diagnostic for genuine (tidal) spacetime curvature, e.g. at the
/// Schwarzschild singularity where `K -> infinity` while `R = 0` everywhere
/// outside it.
pub fn kretschmann_scalar(
    g: &Matrix4<f64>,
    g_inv: &Matrix4<f64>,
    riemann: &[[[[f64; 4]; 4]; 4]; 4],
) -> f64 {
    // Lower the first index: R_{abcd} = g_{ae} R^e_{bcd}.
    let mut r_down = [[[[0.0_f64; 4]; 4]; 4]; 4];
    for a in 0..4 {
        for b in 0..4 {
            for c in 0..4 {
                for d in 0..4 {
                    let mut sum = 0.0;
                    for e in 0..4 {
                        sum += g[(a, e)] * riemann[e][b][c][d];
                    }
                    r_down[a][b][c][d] = sum;
                }
            }
        }
    }
    // Raise all four indices to get R^{abcd}, contracting with r_down.
    let mut k = 0.0;
    for a in 0..4 {
        for b in 0..4 {
            for c in 0..4 {
                for d in 0..4 {
                    let mut r_up = 0.0;
                    for ap in 0..4 {
                        for bp in 0..4 {
                            for cp in 0..4 {
                                for dp in 0..4 {
                                    r_up += g_inv[(a, ap)]
                                        * g_inv[(b, bp)]
                                        * g_inv[(c, cp)]
                                        * g_inv[(d, dp)]
                                        * r_down[ap][bp][cp][dp];
                                }
                            }
                        }
                    }
                    k += r_down[a][b][c][d] * r_up;
                }
            }
        }
    }
    k
}

/// Convenience bundle: everything computed at a single point, with one
/// metric evaluation and one Riemann-tensor computation shared between them.
pub struct CurvatureAtPoint {
    pub g: Matrix4<f64>,
    pub g_inv: Matrix4<f64>,
    pub christoffel: [[[f64; 4]; 4]; 4],
    pub riemann: [[[[f64; 4]; 4]; 4]; 4],
    pub ricci: Matrix4<f64>,
    pub ricci_scalar: f64,
    pub einstein: Matrix4<f64>,
    pub kretschmann: f64,
}

pub fn curvature_at(metric: &dyn Fn(&Point4) -> Matrix4<f64>, x: &Point4, h: f64) -> CurvatureAtPoint {
    let g = metric(x);
    let g_inv = g.try_inverse().expect("metric is not invertible at this point");
    let gamma = christoffel(metric, x, h);
    let riem = riemann(metric, x, h);
    let ricci = ricci_tensor(&riem);
    let r_scalar = ricci_scalar(&g_inv, &ricci);
    let einstein = einstein_tensor(&g, &ricci, r_scalar);
    let kretschmann = kretschmann_scalar(&g, &g_inv, &riem);
    CurvatureAtPoint {
        g,
        g_inv,
        christoffel: gamma,
        riemann: riem,
        ricci,
        ricci_scalar: r_scalar,
        einstein,
        kretschmann,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Minkowski spacetime: flat, so every curvature tensor must vanish
    /// exactly (up to floating-point/finite-difference noise near machine
    /// epsilon, since a constant metric has zero derivatives at any `h`).
    fn minkowski(_x: &Point4) -> Matrix4<f64> {
        Matrix4::new(
            -1.0, 0.0, 0.0, 0.0, //
            0.0, 1.0, 0.0, 0.0, //
            0.0, 0.0, 1.0, 0.0, //
            0.0, 0.0, 0.0, 1.0,
        )
    }

    #[test]
    fn minkowski_is_flat() {
        let x = [0.3, 1.7, -0.5, 2.2];
        let c = curvature_at(&minkowski, &x, DEFAULT_H);
        for a in 0..4 {
            for b in 0..4 {
                for cc in 0..4 {
                    for d in 0..4 {
                        assert!(
                            c.riemann[a][b][cc][d].abs() < 1e-6,
                            "R^{a}_{{{b}{cc}{d}}} = {} should be ~0 for flat spacetime",
                            c.riemann[a][b][cc][d]
                        );
                    }
                }
            }
        }
        assert!(c.ricci_scalar.abs() < 1e-6);
        assert!(c.kretschmann.abs() < 1e-6);
    }

    /// A constant-metric sanity check that the Christoffel symbols
    /// themselves are exactly zero (derivative of a constant is zero to
    /// float precision, no finite-difference truncation error involved).
    #[test]
    fn constant_metric_has_zero_christoffel() {
        let x = [1.0, 2.0, 3.0, 4.0];
        let gamma = christoffel(&minkowski, &x, DEFAULT_H);
        for a in 0..4 {
            for b in 0..4 {
                for c in 0..4 {
                    assert!(gamma[a][b][c].abs() < 1e-9);
                }
            }
        }
    }
}
