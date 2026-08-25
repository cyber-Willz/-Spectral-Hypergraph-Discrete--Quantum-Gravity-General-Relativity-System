//! Ellipsoidal (geodetic) curvature, and its honest relationship to the
//! discrete curvature already computed in `regge.rs`.
//!
//! This module exists because of one real mathematical coincidence, not
//! because GIS and discrete quantum gravity share a domain: both are, at
//! bottom, computing *curvature from a metric*.
//!
//!   - On a smooth surface (the reference ellipsoid), curvature is a
//!     continuous field: the Gaussian curvature `K = 1 / (M · N)`, where `M`
//!     is the meridian radius of curvature and `N` is the prime-vertical
//!     radius of curvature (both functions of latitude and the ellipsoid's
//!     shape parameters).
//!   - On a simplicial complex (`regge.rs`), curvature is concentrated at
//!     hinges as a deficit angle `δ_hinge`, and the discrete
//!     Gauss-Bonnet-like statement is `Σ_hinges δ_hinge · L_hinge = ∫ R √g`
//!     (Regge's theorem, already implemented as `regge_action`).
//!
//! These are the *same object* (integrated scalar curvature) evaluated on
//! two different kinds of discretization of "a curved 2-manifold." This
//! module implements the continuous (ellipsoidal) side and, separately,
//! a literal discrete Gauss-Bonnet check on a closed simplicial 2-surface
//! (icosahedron), so the analogy can be verified numerically rather than
//! asserted.
//!
//! What this module does NOT claim:
//!   - That WGS84 geodesy and the crate's 3D Regge calculus (hyperedges as
//!     tetrahedra) are literally the same computation. They are not: one is
//!     2D (a surface embedded in 3-space with a known analytic curvature
//!     formula), the other is 3D intrinsic Regge calculus with no ambient
//!     embedding. The bridge here is conceptual and is only asserted at the
//!     level that is actually checked below (discrete Gauss-Bonnet on a
//!     closed 2-surface reproducing the sphere's total curvature 4π).
//!   - Any claim about geodetic datums, projections, or coordinate
//!     transforms being relevant to quantum gravity. They are not; they are
//!     omitted from this crate on purpose (see README).

use std::f64::consts::PI;

/// A reference ellipsoid: semi-major axis `a`, flattening `f`.
#[derive(Debug, Clone, Copy)]
pub struct Ellipsoid {
    pub a: f64,
    pub f: f64,
}

impl Ellipsoid {
    /// WGS84 parameters (the datum GPS and most web maps use).
    pub fn wgs84() -> Self {
        Ellipsoid {
            a: 6_378_137.0,
            f: 1.0 / 298.257_223_563,
        }
    }

    pub fn b(&self) -> f64 {
        self.a * (1.0 - self.f)
    }

    /// First eccentricity squared, e² = 2f - f².
    pub fn e2(&self) -> f64 {
        2.0 * self.f - self.f * self.f
    }

    /// Meridian radius of curvature M(φ): curvature in the north-south
    /// direction at geodetic latitude φ (radians).
    pub fn meridian_radius(&self, lat_rad: f64) -> f64 {
        let e2 = self.e2();
        let sin_phi = lat_rad.sin();
        self.a * (1.0 - e2) / (1.0 - e2 * sin_phi * sin_phi).powf(1.5)
    }

    /// Prime-vertical radius of curvature N(φ): curvature in the east-west
    /// direction at geodetic latitude φ (radians).
    pub fn prime_vertical_radius(&self, lat_rad: f64) -> f64 {
        let e2 = self.e2();
        let sin_phi = lat_rad.sin();
        self.a / (1.0 - e2 * sin_phi * sin_phi).sqrt()
    }

    /// Gaussian curvature K(φ) = 1 / (M·N) of the ellipsoid at latitude φ.
    /// This is the exact continuous analogue of a Regge deficit-angle
    /// density: both are "curvature per unit area" of a 2-metric.
    pub fn gaussian_curvature(&self, lat_rad: f64) -> f64 {
        1.0 / (self.meridian_radius(lat_rad) * self.prime_vertical_radius(lat_rad))
    }
}

/// Discrete Gauss-Bonnet check on a closed simplicial 2-surface: for any
/// closed genus-0 triangulated surface, Σ_vertices (2π - angle_sum(v)) = 4π.
/// This is the literal 2D analogue of the 3D Regge identity already
/// verified in `regge.rs`/`regge_tests.rs` (flat-cube deficit-angle test),
/// and is what actually justifies treating "ellipsoidal curvature" and
/// "Regge deficit-angle curvature" as the same concept rather than a loose
/// verbal analogy.
///
/// `vertex_angle_sums[v]` is the sum of triangle angles meeting at vertex v
/// (radians). Returns the total discrete curvature Σ (2π - angle_sum(v)).
pub fn discrete_gauss_bonnet_total(vertex_angle_sums: &[f64]) -> f64 {
    vertex_angle_sums.iter().map(|&s| 2.0 * PI - s).sum()
}

/// Regular icosahedron: 12 vertices, each surrounded by 5 equilateral
/// triangles (angle sum 5·60° = 300° = 5π/3), used as the standard
/// verification case for discrete Gauss-Bonnet (Σ deficit = 4π exactly,
/// independent of edge length, matching Euler characteristic χ=2 of a
/// sphere via Σδ = 2πχ).
pub fn icosahedron_vertex_angle_sums() -> Vec<f64> {
    vec![5.0 * (PI / 3.0); 12]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wgs84_curvature_is_positive_and_within_known_range() {
        let e = Ellipsoid::wgs84();
        for lat_deg in [0.0f64, 15.0, 30.0, 45.0, 60.0, 75.0, 89.9] {
            let k = e.gaussian_curvature(lat_deg.to_radians());
            assert!(k > 0.0, "Gaussian curvature of an ellipsoid is everywhere positive");
            // K = 1/(M*N); for Earth, M and N are both O(6.36e6 - 6.40e6 m),
            // so K should sit within a tight, known band (~2.4e-14 m^-2).
            assert!(
                k > 2.3e-14 && k < 2.5e-14,
                "K={k} out of expected WGS84 range at lat {lat_deg}"
            );
        }
    }

    #[test]
    fn meridian_and_prime_vertical_radii_match_known_pole_equator_values() {
        let e = Ellipsoid::wgs84();
        // At the equator, N = a exactly.
        let n_eq = e.prime_vertical_radius(0.0);
        assert!((n_eq - e.a).abs() < 1e-6);
        // At the pole, M = N = a / sqrt(1 - e^2) = a^2/b (the polar radius
        // of curvature), a standard closed-form geodesy identity.
        let m_pole = e.meridian_radius(PI / 2.0);
        let n_pole = e.prime_vertical_radius(PI / 2.0);
        let polar_roc = e.a * e.a / e.b();
        assert!((m_pole - polar_roc).abs() / polar_roc < 1e-9);
        assert!((n_pole - polar_roc).abs() / polar_roc < 1e-9);
    }

    #[test]
    fn spherical_limit_recovers_constant_curvature_1_over_r2() {
        // f -> 0 degenerates the ellipsoid to a sphere of radius a; K should
        // become the classical constant 1/a^2, independent of latitude.
        let sphere = Ellipsoid { a: 6_371_000.0, f: 0.0 };
        let k_eq = sphere.gaussian_curvature(0.0);
        let k_45 = sphere.gaussian_curvature(45f64.to_radians());
        let k_pole = sphere.gaussian_curvature(89.999f64.to_radians());
        let expected = 1.0 / (sphere.a * sphere.a);
        for k in [k_eq, k_45, k_pole] {
            assert!((k - expected).abs() / expected < 1e-9);
        }
    }

    #[test]
    fn discrete_gauss_bonnet_on_icosahedron_gives_4pi() {
        // The literal 2D discrete-curvature analogue of the 3D Regge
        // identity this crate already verifies in regge_tests.rs.
        let total = discrete_gauss_bonnet_total(&icosahedron_vertex_angle_sums());
        assert!((total - 4.0 * PI).abs() < 1e-9);
    }
}
