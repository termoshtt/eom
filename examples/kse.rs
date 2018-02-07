extern crate eom;
extern crate ndarray;
extern crate ndarray_linalg;

use ndarray::*;
use ndarray_linalg::*;
use eom::*;
use eom::traits::*;

fn main() {
    let n = 256;
    let l = 100.0;
    let dt = 1e-3;

    let eom = pde::KSE::new(n, l);
    let mut eom2 = eom.clone();
    let n_coef = eom.model_size();
    let mut teo = semi_implicit::DiagRK4::new(eom, dt);

    let x0: Array1<c64> = c64::new(0.01, 0.0) * random(n_coef);

    let ts = adaptor::time_series(x0, &mut teo);

    let end_time = 100_000;
    let interval = 1000;
    for (t, v) in ts.take(end_time).enumerate() {
        if t % interval != 0 {
            continue;
        }
        print!("{:e}", dt * t as f64);
        let u = eom2.convert_u(v.as_slice().unwrap());
        for val in u.iter() {
            print!(",{:e}", val);
        }
        println!("");
    }
}
