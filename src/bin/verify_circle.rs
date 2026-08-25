use spectral_dqg::continuum_limit::random_simple_regular_graph;
use spectral_dqg::nonbacktracking::hashimoto_matrix;
use spectral_dqg::bulk_spacing::complex_eigenvalues;

fn main() {
    let d = 4usize;
    let g = random_simple_regular_graph(120, d, 3, 20000);
    let (b, _) = hashimoto_matrix(&g);
    let eigs = complex_eigenvalues(&b).unwrap();
    let expected_radius = ((d - 1) as f64).sqrt();

    let nontrivial: Vec<_> = eigs.iter()
        .filter(|z| !((z.re - 1.0).abs() < 1e-6 && z.im.abs() < 1e-6))
        .filter(|z| !((z.re + 1.0).abs() < 1e-6 && z.im.abs() < 1e-6))
        .collect();

    let radii: Vec<f64> = nontrivial.iter().map(|z| z.re.hypot(z.im)).collect();
    let mean_radius = radii.iter().sum::<f64>() / radii.len() as f64;
    let max_dev = radii.iter().map(|r| (r - expected_radius).abs()).fold(0.0_f64, f64::max);
    let frac_within_1pct = radii.iter().filter(|r| ((*r - expected_radius).abs() / expected_radius) < 0.01).count();

    println!("expected radius sqrt(d-1) = {expected_radius:.6}");
    println!("n_nontrivial = {}, mean radius = {mean_radius:.6}, max deviation = {max_dev:.6}", nontrivial.len());
    println!("fraction within 1% of expected radius: {:.4}", frac_within_1pct as f64 / radii.len() as f64);

    // print histogram of radii to see how tight
    let mut sorted = radii.clone();
    sorted.sort_by(|a,b| a.partial_cmp(b).unwrap());
    println!("min={:.6} p10={:.6} p50={:.6} p90={:.6} max={:.6}",
        sorted[0], sorted[sorted.len()/10], sorted[sorted.len()/2], sorted[sorted.len()*9/10], sorted[sorted.len()-1]);
}
