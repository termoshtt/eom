
extern crate ndarray;
extern crate ndarray_odeint;
extern crate itertools;

use ndarray::rcarr1;
use itertools::iterate;
use ndarray_odeint::prelude::*;

fn main() {
    let dt = 0.01;
    let eom = Lorenz63::default();
    let teo = semi_implicit::diag_rk4(eom, dt);
    let ts = iterate(rcarr1(&[1.0, 0.0, 0.0]), |y| teo.iterate(y.clone()));
    let end_time = 10000;
    println!("time,x,y,z");
    for (t, v) in ts.take(end_time).enumerate() {
        println!("{},{},{},{}", t as f64 * dt, v[0], v[1], v[2]);
    }
}
