
extern crate ndarray;
extern crate ndarray_odeint;
extern crate itertools;
extern crate simple_complex;
extern crate num_traits;

use ndarray::rcarr1;
use itertools::iterate;
use ndarray_odeint::prelude::*;
use simple_complex::c64;
use num_traits::Zero;

fn main() {
    let dt = 1e-5;
    let eom = GoyShell::default();
    let teo = semi_implicit::diag_rk4(eom, dt);
    let mut x0 = rcarr1(&vec![c64::zero(); 27]);
    x0[2] = c64::new(1.0, 0.0);
    x0[3] = c64::new(1.0, 0.0);
    x0[4] = c64::new(1.0, 0.0);
    x0[5] = c64::new(1.0, 0.0);
    x0[6] = c64::new(1.0, 0.0);
    let ts = iterate(x0, |y| teo.iterate(y.clone()));
    let end_time = 100_000_000;
    let interval = 100;
    print!("time");
    for i in 0..27 {
        print!(",r{},c{}", i, i);
    }
    println!("");
    for (t, v) in ts.take(end_time).enumerate() {
        if t % interval != 0 {
            continue;
        }
        print!("{:e}", dt * t as f64);
        for c in v.iter() {
            print!(",{:e},{:e}", c.re(), c.im());
        }
        println!("");
    }
}