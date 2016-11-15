
extern crate ndarray;
extern crate ndarray_odeint;

use ndarray::prelude::*;

fn main() {
    let dt = 0.01;
    let l = |y| ndarray_odeint::lorenz63(10.0, 28.0, 8.0 / 3.0, y);
    let ts = ndarray_odeint::TimeSeries {
        teo: |y| ndarray_odeint::rk4(&l, dt, y),
        state: arr1(&[1.0, 0.0, 0.0]),
    };
    let end_time = 10000;
    println!("time,x,y,z");
    for (t, v) in ts.take(end_time).enumerate() {
        println!("{},{},{},{}", t as f64 * dt, v[0], v[1], v[2]);
    }
}
