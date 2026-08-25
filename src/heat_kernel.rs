//! Heat-kernel trace P(t) = Tr(e^{-tΔ}) = Σ_j e^{-t λ_j}, and the running
//! spectral dimension
//!
//! ```text
//!     d_s(t) = -2 · d ln P(t) / d ln t
//! ```
//!
//! estimated by a centered finite difference in log-t. This is exactly the
//! estimator used for causal dynamical triangulations / causal-set spectral
//! dimension in the physics literature — it is *not* a single number but a
//! function of scale, and for any finite graph it has two well-known
//! artifacts worth stating up front rather than hiding:
//!
//!   * IR end (t → ∞): P(t) → (number of connected components), so
//!     d_s(t) → 0. This is a real finite-size effect, not a bug.
//!   * UV end (t → 0): P(t) → N (number of vertices), so d_s(t) → 0 as well
//!     once t is smaller than the inverse of the largest eigenvalue — a
//!     finite graph has a UV lattice cutoff, exactly as the write-up's
//!     "Planckian scale" language suggests it should.
//!
//! The physically meaningful part is the plateau in between, if one exists.

pub fn heat_trace(eigenvalues: &[f64], t: f64) -> f64 {
    eigenvalues.iter().map(|&lam| (-t * lam).exp()).sum()
}

pub struct SpectralDimensionPoint {
    pub t: f64,
    pub p_t: f64,
    pub d_s: f64,
}

/// Sweep t geometrically over [t_min, t_max] and estimate d_s(t) via a
/// centered log-log finite difference at each interior sample.
pub fn spectral_dimension_flow(
    eigenvalues: &[f64],
    t_min: f64,
    t_max: f64,
    n_samples: usize,
) -> Vec<SpectralDimensionPoint> {
    assert!(n_samples >= 3);
    let log_min = t_min.ln();
    let log_max = t_max.ln();
    let ts: Vec<f64> = (0..n_samples)
        .map(|i| {
            let frac = i as f64 / (n_samples as f64 - 1.0);
            (log_min + frac * (log_max - log_min)).exp()
        })
        .collect();
    let ps: Vec<f64> = ts.iter().map(|&t| heat_trace(eigenvalues, t)).collect();

    let mut out = Vec::with_capacity(n_samples - 2);
    for i in 1..n_samples - 1 {
        let d_ln_p = ps[i + 1].ln() - ps[i - 1].ln();
        let d_ln_t = ts[i + 1].ln() - ts[i - 1].ln();
        let d_s = -2.0 * d_ln_p / d_ln_t;
        out.push(SpectralDimensionPoint {
            t: ts[i],
            p_t: ps[i],
            d_s,
        });
    }
    out
}
