//! Calculate Covariant Lyapunov Vector
//!
//! This function consumes much memory since this saves matrices duraing the time evolution.

extern crate eom;
extern crate ndarray;
extern crate ndarray_linalg;

use std::io::Write;
use ndarray::*;
use ndarray_linalg::*;

use eom::*;
use eom::traits::*;

fn main() {
    let dt = 0.01;
    let eom = ode::Lorenz63::default();
    let teo = explicit::RK4::new(eom, dt);
    let duration = 100000;
    let ts = lyapunov::vectors(teo, arr1(&[1.0, 0.0, 0.0]), 1e-7, duration);
    let mut l = Array::zeros(3);
    println!("v0v1, v0v2, v1v2");
    for (_x, v, f) in ts.into_iter().rev() {
        let v0 = v.axis_iter(Axis(1)).nth(0).unwrap();
        let v1 = v.axis_iter(Axis(1)).nth(1).unwrap();
        let v2 = v.axis_iter(Axis(1)).nth(2).unwrap();
        println!("{}, {}, {}", v0.dot(&v1), v0.dot(&v2), v1.dot(&v2));
        l += &f.map(|x| x.abs().ln());
    }
    write!(
        &mut std::io::stderr(),
        "exponents = {:?}\n",
        l / (dt * duration as f64)
    ).unwrap();
}
