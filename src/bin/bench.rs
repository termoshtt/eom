
extern crate ndarray;
extern crate ndarray_odeint;

use ndarray::*;
use ndarray_odeint::*;

fn main() {
    let dt = 0.01;
    let eom = model::Lorenz63::default();
    let teo = explicit::rk4(eom, dt);
    let mut x = arr1(&[1.0, 0.0, 0.0]);
    for _ in 0..100_000_000 {
        teo.iterate(&mut x);
    }
}
