
extern crate ndarray;
extern crate ndarray_odeint as odeint;

use ndarray::prelude::*;

fn main() {
    let l = |y| odeint::lorenz63(10.0, 28.0, 8.0 / 3.0, y);
    let ts = odeint::TimeSeries {
        teo: |y| odeint::rk4(&l, 0.01, y),
        state: arr1(&[1.0, 0.0, 0.0]),
    };
    let end_time = 10000;
    for v in ts.take(end_time) {
        println!("{} {} {}", v[0], v[1], v[2]);
    }
}
