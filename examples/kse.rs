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
    let interval = 1000;
    let step = 1000;

    let eom = pde::KSE::new(n, l);
    let mut eom2 = eom.clone();
    let n_coef = eom.model_size();
    let teo = semi_implicit::DiagRK4::new(eom, dt);
    let mut teo = adaptor::nstep(teo, interval);

    let x: Array1<c64> = c64::new(0.01, 0.0) * random(n_coef);
    let x = adaptor::iterate(&mut teo, x, 100);

    let ts = adaptor::time_series(x, &mut teo);
    for (t, v) in ts.take(step).enumerate() {
        let time = dt * t as f64;
        print!("{:e},", time);
        let u = eom2.convert_u(v.as_slice().unwrap());
        let nums: Vec<_> = u.iter().map(|x| format!("{:e}", x)).collect();
        println!("{}", nums.join(","));
    }
}
