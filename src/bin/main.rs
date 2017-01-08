
extern crate ndarray;
extern crate ndarray_odeint;
extern crate itertools;

use ndarray::prelude::*;
use ndarray::rcarr1;
use itertools::iterate;
use ndarray_odeint::lorenz63 as l63;
use ndarray_odeint::traits::*;
use ndarray_odeint::explicit::*;

fn main() {
    let dt = 0.01;
    let eom = l63::Lorenz63::default();
    let euler = Euler::new(eom, dt);
    let ts = iterate(rcarr1(&[1.0, 0.0, 0.0]), |y| euler.iterate(y.clone()));
    let end_time = 10000;
    println!("time,x,y,z");
    for (t, v) in ts.take(end_time).enumerate() {
        println!("{},{},{},{}", t as f64 * dt, v[0], v[1], v[2]);
    }
}
