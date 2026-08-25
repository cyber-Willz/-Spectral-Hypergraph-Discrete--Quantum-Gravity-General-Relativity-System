//! GNSS relativistic clock corrections.
//!
//! Of everything in the GIS write-up, this is the one piece with a genuine,
//! direct physics link to a quantum-gravity crate rather than a shared-math
//! coincidence: GPS satellite clocks require both a special-relativistic
//! (velocity time dilation) and a general-relativistic (gravitational
//! potential) correction, and the net effect is a textbook worked example
//! of weak-field GR — the same theory whose discrete/path-integral treatment
//! (Regge calculus) is this crate's actual subject.
//!
//! This module computes both corrections from first principles (orbital
//! mechanics + Schwarzschild weak-field potential) and checks the result
//! against the well-known net figure of ~38 microseconds/day fast, which is
//! the number every GPS/relativity textbook quotes and is the standard
//! sanity check for this calculation.
//!
//! What this module does NOT claim:
//!   - Any connection to this crate's discrete Regge action or path
//!     integral. The correction below is a standard continuum GR
//!     perturbative calculation (weak-field metric, first-order velocity
//!     and potential terms); it is not derived from or fed into the
//!     Regge/spectral machinery elsewhere in this crate. It is included
//!     because it is the one physically real, textbook-verifiable link
//!     between "GIS" and "GR," not because it exercises this crate's core
//!     algorithms.

const G: f64 = 6.674_30e-11; // gravitational constant, m^3 kg^-1 s^-2
const M_EARTH: f64 = 5.972_2e24; // kg
pub const R_EARTH: f64 = 6_371_000.0; // m, mean radius, exposed for callers
const C: f64 = 299_792_458.0; // m/s

/// Special-relativistic time dilation fractional rate for a satellite in a
/// circular orbit of radius `r` (from Earth's center): the moving clock
/// runs SLOW relative to a fixed clock at Earth's center by
/// `-v²/(2c²) = -GM/(2 r c²)` (circular-orbit speed v² = GM/r).
/// Returns the fractional rate (dimensionless, seconds of drift per second
/// of proper time) — negative means "runs slow."
pub fn special_relativistic_rate(orbital_radius_m: f64) -> f64 {
    -(G * M_EARTH) / (2.0 * orbital_radius_m * C * C)
}

/// General-relativistic (gravitational potential) fractional rate: a clock
/// higher in Earth's gravity well runs FAST relative to a clock at Earth's
/// surface by `GM/c² · (1/r_surface - 1/r_orbit)`.
pub fn general_relativistic_rate(orbital_radius_m: f64, reference_radius_m: f64) -> f64 {
    (G * M_EARTH / (C * C)) * (1.0 / reference_radius_m - 1.0 / orbital_radius_m)
}

/// Net fractional clock rate for a GPS satellite (orbital radius ~26,560 km
/// from Earth's center for the standard ~20,180 km altitude), relative to a
/// clock at Earth's mean radius, combining both effects.
pub fn net_gps_relativistic_rate(orbital_radius_m: f64, reference_radius_m: f64) -> f64 {
    special_relativistic_rate(orbital_radius_m) + general_relativistic_rate(orbital_radius_m, reference_radius_m)
}

/// Convert a fractional rate to microseconds of drift accumulated per day.
pub fn rate_to_microseconds_per_day(fractional_rate: f64) -> f64 {
    fractional_rate * 86_400.0 * 1.0e6
}

#[cfg(test)]
mod tests {
    use super::*;

    const GPS_ORBITAL_RADIUS_M: f64 = 26_560_000.0; // ~20,180 km altitude + R_EARTH

    #[test]
    fn special_relativity_effect_makes_satellite_clock_run_slow() {
        let rate = special_relativistic_rate(GPS_ORBITAL_RADIUS_M);
        assert!(rate < 0.0, "velocity time dilation must slow the satellite clock");
        let us_per_day = rate_to_microseconds_per_day(rate);
        // Textbook value: SR effect is about -7 microseconds/day.
        assert!(
            (us_per_day + 7.0).abs() < 1.0,
            "expected ~-7 us/day from SR, got {us_per_day}"
        );
    }

    #[test]
    fn general_relativity_effect_makes_satellite_clock_run_fast() {
        let rate = general_relativistic_rate(GPS_ORBITAL_RADIUS_M, R_EARTH);
        assert!(rate > 0.0, "weaker gravity at altitude must speed up the satellite clock");
        let us_per_day = rate_to_microseconds_per_day(rate);
        // Textbook value: GR effect is about +45 microseconds/day.
        assert!(
            (us_per_day - 45.0).abs() < 2.0,
            "expected ~+45 us/day from GR, got {us_per_day}"
        );
    }

    #[test]
    fn net_effect_matches_the_well_known_38_microseconds_per_day() {
        let net = net_gps_relativistic_rate(GPS_ORBITAL_RADIUS_M, R_EARTH);
        let us_per_day = rate_to_microseconds_per_day(net);
        // The GIS write-up cites ~38 us/day net; this is the standard
        // figure quoted in every treatment of GPS relativistic corrections
        // (Ashby 2003 and countless textbooks since).
        assert!(
            (us_per_day - 38.0).abs() < 2.0,
            "expected net ~+38 us/day, got {us_per_day}"
        );
    }
}
