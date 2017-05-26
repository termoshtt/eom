
extern crate ndarray;
extern crate ndarray_odeint;
extern crate ndarray_linalg;
extern crate itertools;

use std::io::Write;
use ndarray::*;
use ndarray_odeint::prelude::*;
use ndarray_odeint::lyapunov::*;

fn main() {
    let dt = 0.01;
    let eom = Lorenz63::default();
    let teo = explicit::RK4::new(eom, dt);
    let duration = 100000;
    let ts = clv(&teo, rcarr1(&[1.0, 0.0, 0.0]), 1e-7, duration);
    let mut l = Array::zeros(3);
    println!("v0v1, v0v2, v1v2");
    for (_, v, f) in ts.into_iter().rev() {
        let v0 = v.axis_iter(Axis(1)).nth(0).unwrap();
        let v1 = v.axis_iter(Axis(1)).nth(1).unwrap();
        let v2 = v.axis_iter(Axis(1)).nth(2).unwrap();
        println!("{}, {}, {}", v0.dot(&v1), v0.dot(&v2), v1.dot(&v2));
        l += &f.map(|x| x.abs().ln());
    }
    write!(&mut std::io::stderr(),
           "exponents = {:?}\n",
           l / (dt * duration as f64))
        .unwrap();
}
