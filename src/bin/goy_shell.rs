
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
    let x0 = rcarr1(&vec![c64::zero(); 27]);
    let ts = iterate(x0, |y| teo.iterate(y.clone()));
    let end_time = 100;
    print!("r0,c0");
    for i in 1..27 {
        print!(",r{},c{}", i, i);
    }
    println!("");
    for (_, v) in ts.take(end_time).enumerate() {
        for (i, c) in v.iter().enumerate() {
            if i == 0 {
                print!("{},{}", c.re(), c.im());
            } else {
                print!(",{},{}", c.re(), c.im());
            }
        }
        println!("");
    }
}
