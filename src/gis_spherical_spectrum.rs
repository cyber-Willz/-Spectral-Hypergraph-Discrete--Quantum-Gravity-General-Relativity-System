//! Spherical-harmonic spectrum of the Laplace-Beltrami operator on S²,
//! reusing this crate's existing heat-trace machinery.
//!
//! Geodetic gravity models (e.g. EGM2008, referenced in the GIS write-up)
//! expand Earth's gravity potential in spherical harmonics — a Fourier-like
//! basis on the sphere whose Laplace-Beltrami eigenvalues are the classical
//! closed form
//!
//! ```text
//! λ_l = l(l+1) / R^2,   with degeneracy (2l+1) for each degree l.
//! ```
//!
//! That is *exactly* the kind of object `heat_kernel::heat_trace` already
//! consumes (a list of Laplacian eigenvalues) — so the honest way to
//! connect this crate's spectral-graph-theory core to the GIS write-up's
//! spherical-harmonic material is to generate the sphere's true spectrum
//! and feed it through the same heat-trace/spectral-dimension code already
//! used for hypergraph Laplacians, rather than writing a parallel
//! implementation.
//!
//! What this module does NOT claim:
//!   - That a discrete hypergraph Laplacian spectrum "converges to" the
//!     sphere spectrum in any specific limit. No such convergence is proven
//!     or numerically demonstrated here; only that both are valid inputs to
//!     the same heat-trace formalism, and that this continuum spectrum
//!     produces the textbook-known spectral dimension d_s = 2 (a 2-manifold)
//!     in the plateau region, which then serves as a sanity reference point
//!     for the finite-graph d_s(t) plots already produced elsewhere in this
//!     crate.
//!   - Any actual EGM2008 gravity coefficients; only the eigenvalue
//!     structure (degree, degeneracy) is used, not real geophysical data.

/// Laplace-Beltrami eigenvalues on a sphere of radius R, up to and including
/// degree `l_max`, each repeated according to its (2l+1)-fold degeneracy —
/// i.e. exactly the eigenvalue *multiset* that a real spherical-harmonic
/// gravity-field expansion truncated at degree l_max would carry.
pub fn sphere_laplacian_eigenvalues(radius: f64, l_max: usize) -> Vec<f64> {
    let mut eigs = Vec::with_capacity((l_max + 1) * (l_max + 1));
    for l in 0..=l_max {
        let lambda = (l * (l + 1)) as f64 / (radius * radius);
        let degeneracy = 2 * l + 1;
        eigs.extend(std::iter::repeat(lambda).take(degeneracy));
    }
    eigs
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::heat_kernel::{heat_trace, spectral_dimension_flow};

    #[test]
    fn eigenvalue_count_matches_l_max_plus_one_squared() {
        let l_max = 10;
        let eigs = sphere_laplacian_eigenvalues(1.0, l_max);
        assert_eq!(eigs.len(), (l_max + 1) * (l_max + 1));
    }

    #[test]
    fn zero_mode_is_present_and_unique_degree_zero_eigenvalue_is_zero() {
        let eigs = sphere_laplacian_eigenvalues(6_371_000.0, 4);
        assert!((eigs[0]).abs() < 1e-12, "l=0 eigenvalue must be exactly 0");
    }

    #[test]
    fn heat_trace_at_t_zero_equals_total_multiplicity() {
        let l_max = 8;
        let eigs = sphere_laplacian_eigenvalues(1.0, l_max);
        let p0 = heat_trace(&eigs, 0.0);
        assert!((p0 - eigs.len() as f64).abs() < 1e-9);
    }

    #[test]
    fn spectral_dimension_plateau_is_near_2_for_a_2_manifold() {
        // Reusing this crate's own spectral_dimension_flow estimator: for a
        // genuine 2-manifold's Laplacian spectrum, d_s(t) should sit near 2
        // in its plateau region (away from the UV/IR finite-truncation
        // artifacts this crate's heat_kernel.rs already documents).
        let l_max = 60;
        let eigs = sphere_laplacian_eigenvalues(1.0, l_max);
        let flow = spectral_dimension_flow(&eigs, 1e-3, 1e-1, 25);
        let mid = &flow[flow.len() / 2];
        assert!(
            (mid.d_s - 2.0).abs() < 0.3,
            "expected d_s near 2 in plateau, got {} at t={}",
            mid.d_s,
            mid.t
        );
    }
}
