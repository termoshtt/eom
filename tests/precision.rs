
extern crate ndarray;
extern crate ndarray_odeint;
extern crate ndarray_linalg;
extern crate itertools;
extern crate num_traits;

use std::fs::*;
use std::io::Write;
use ndarray::prelude::*;
use ndarray_linalg::prelude::*;
use itertools::iterate;
use ndarray_odeint::lorenz63 as l63;
use num_traits::int::PrimInt;

type V = Array<f64, Ix1>;

macro_rules! impl_precision_test {
    ($name:ident, $method:path, $filename:expr) => {
#[test]
fn $name() {
    let p = l63::default_parameter();
    let l = |y| l63::f(p, y);
    let data: Vec<_> = (0..12)
        .map(|n| {
            let dt = 0.1 / 2.pow(n) as f64;
            let t = 100 * 2.pow(n);
            let ts = iterate(arr1(&[1.0, 0.0, 0.0]), |y| $method(&l, dt, y.clone()));
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
}} // impl_precision_test

impl_precision_test!(euler, ndarray_odeint::explicit::euler, "euler.csv");
impl_precision_test!(rk4, ndarray_odeint::explicit::rk4, "rk4.csv");
