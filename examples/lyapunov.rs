//! Lyapunov Analysis example

use eom::*;
use ndarray::*;

fn main() {
    let dt = 0.01;
    let eom = ode::Lorenz63::default();
    let teo = explicit::RK4::new(eom, dt);
    let l = lyapunov::exponents(teo, arr1(&[1.0, 0.0, 0.0]), 1e-7, 100000);
    println!("Lyapunov Exponents:");
    println!("- l0 = {}", l[0]);
    println!("- l1 = {}", l[1]);
    println!("- l2 = {}", l[2]);
}
