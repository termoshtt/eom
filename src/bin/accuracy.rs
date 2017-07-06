
extern crate ndarray;
extern crate ndarray_odeint;
extern crate ndarray_linalg;
extern crate num_traits;

use std::fs::*;
use std::io::Write;
use ndarray::*;
use ndarray_odeint::*;
use ndarray_linalg::*;
use num_traits::int::PrimInt;

macro_rules! impl_accuracy {
    ($name:ident, $method:path, $filename:expr) => {
fn $name() {
    let data: Vec<_> = (0..12)
        .map(|n| {
            let dt = 0.1 / 2.pow(n) as f64;
            let eom = model::Lorenz63::default();
            let teo = $method(eom, dt);
            let t = 100 * 2.pow(n);
            let ts = TimeSeries::new(rcarr1(&[1.0, 0.0, 0.0]), &teo);
            (dt, ts.take(t+1).last().unwrap())
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

impl_accuracy!(euler, explicit::euler, "euler.csv");
impl_accuracy!(heun, explicit::heun, "heun.csv");
impl_accuracy!(rk4, explicit::rk4, "rk4.csv");
impl_accuracy!(diag_rk4, semi_implicit::diag_rk4, "diag_rk4.csv");

fn main() {
    euler();
    heun();
    rk4();
    diag_rk4();
}
