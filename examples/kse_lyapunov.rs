extern crate eom;
extern crate ndarray;
extern crate ndarray_linalg;

use ndarray::*;
use ndarray_linalg::*;
use eom::*;
use eom::traits::*;

fn main() {
    let n = 128;
    let l = 100.0;
    let dt = 1e-3;

    let eom = pde::KSE::new(n, l);
    let n_coef = eom.model_size();
    let teo = semi_implicit::diag_rk4(eom, dt);
    let mut teo = adaptor::nstep(teo, 100); // every t=0.1

    eprint!("Initialing... ");
    let x: Array1<c64> = c64::new(0.01, 0.0) * random(n_coef);
    let x = adaptor::time_series(x, &mut teo).take(100).last().unwrap(); // drop first 10.0
    eprintln!("Done!");

    eprint!("Start Lyapunov iteration... ");
    let mut l: Array1<f64> = Array::zeros(n_coef);
    for (t, (_x, _q, r)) in lyapunov::Series::new(teo, x, 1e-7).take(100000).enumerate() {
        let d = r.diag().map(|x| x.abs().ln());
        azip!(mut l, d in { *l += d } );
        let nums: Vec<_> = l.iter()
            .map(|x| format!("{:.07}", x / (t + 1) as f64))
            .collect();
        println!("{}", nums.join(","));
    }
    eprintln!("Finished");
}
