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
    let x0: Array1<c64> = c64::new(0.01, 0.0) * random(n_coef);
    let l = lyapunov::exponents(teo, x0, 1e-7, 100000);
    println!("{}", l);
}
