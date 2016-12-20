
extern crate ndarray;
extern crate ndarray_odeint;
extern crate itertools;

use ndarray::prelude::*;
use itertools::iterate;
use ndarray_odeint::lorenz63 as l63;

fn main() {
    let dt = 0.01;
    let p = l63::default_parameter();
    let l = |y| l63::f(p, y);
    let ts = iterate(arr1(&[1.0, 0.0, 0.0]),
                     |y| ndarray_odeint::explicit::rk4(&l, dt, y.clone()));
    let end_time = 10000;
    println!("time,x,y,z");
    for (t, v) in ts.take(end_time).enumerate() {
        println!("{},{},{},{}", t as f64 * dt, v[0], v[1], v[2]);
    }
}
