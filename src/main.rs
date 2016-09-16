
extern crate ndarray;
extern crate ndarray_odeint;

use ndarray::prelude::*;

fn main() {
    let mut x = arr1(&[1.0, 0.0, 0.0]);
    let l = |y| ndarray_odeint::lorenz63(10., 28., 8.0 / 3.0, y);
    let teo = |y| ndarray_odeint::rk4(&l, 0.01, y);
    for _ in 0..10000000 {
        x = teo(x);
    }
    println!("{:?}", x);
}
