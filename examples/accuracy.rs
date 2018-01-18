extern crate eom;
extern crate ndarray;
extern crate ndarray_linalg;
extern crate num_traits;

use std::fs::*;
use std::io::Write;
use ndarray::*;
use eom::*;
use ndarray_linalg::*;
use num_traits::int::PrimInt;

macro_rules! impl_accuracy {
    ($name:ident, $method:path, $filename:expr) => {
fn $name() {
    let data: Vec<_> = (0..12)
        .map(|n| {
            let dt = 0.1 / 2.pow(n) as f64;
            let eom = ode::Lorenz63::default();
            let mut teo = $method(eom, dt);
            let t = 100 * 2.pow(n);
            let ts = adaptor::time_series(rcarr1(&[1.0, 0.0, 0.0]), &mut teo);
            (dt, ts.take(t).last().unwrap())
        })
        .collect();

    let mut f = File::create($filename).unwrap();
    write!(&mut f, "dt,dev\n").unwrap();
    for i in 0..(data.len() - 1) {
        let dt = data[i].0;
        let dev = (&data[i + 1].1 - &data[i].1).norm();
        write!(&mut f, "{},{}\n", dt, dev).unwrap();
    }
}
}} // impl_accuracy_test

impl_accuracy!(euler, explicit::Euler::new, "euler.csv");
impl_accuracy!(heun, explicit::Heun::new, "heun.csv");
impl_accuracy!(rk4, explicit::RK4::new, "rk4.csv");
impl_accuracy!(diag_rk4, semi_implicit::diag_rk4, "diag_rk4.csv");

fn main() {
    euler();
    heun();
    rk4();
    diag_rk4();
}
