
extern crate ndarray;
extern crate ndarray_odeint;
extern crate ndarray_linalg;
extern crate itertools;

use ndarray::*;
use ndarray_linalg::prelude::*;
use ndarray_odeint::prelude::*;
use ndarray_odeint::lyapunov::*;
use itertools::iterate;

fn main() {
    let dt = 0.01;
    let eom = Lorenz63::default();
    let teo = explicit::rk4(eom, dt);
    let ts = iterate(rcarr1(&[1.0, 0.0, 0.0]), |y| teo.iterate(y.clone()));
    let qr_series = ts.scan(Array::eye(3), |st, x| {
        let j = teo.jacobian(x.clone(), 1e-7);
        let (q_next, r) = j.dot(st).qr().unwrap();
        let q = std::mem::replace(st, q_next);
        Some((q, r))
    });
    let duration = 100000;
    let mut l = arr1(&[0.0, 0.0, 0.0]);
    for (_, r) in qr_series.skip(duration / 10).take(duration) {
        l = l + r.diag().map(|x| x.abs().ln());
    }
    println!("{:?}", l / (dt * duration as f64));
}
