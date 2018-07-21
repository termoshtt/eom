extern crate eom;
extern crate ndarray;

use eom::traits::*;
use eom::*;
use ndarray::*;

fn main() {
    let dt = 0.01;
    let eom = ode::Lorenz63::default();
    let mut teo = explicit::RK4::new(eom, dt);
    let mut x: Array1<f64> = arr1(&[1.0, 0.0, 0.0]);
    for _ in 0..100_000_000 {
        teo.iterate(&mut x);
    }
}
