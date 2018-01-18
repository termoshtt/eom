extern crate eom;
extern crate ndarray;
extern crate num_complex;
extern crate num_traits;

use ndarray::rcarr1;
use eom::*;
use num_complex::Complex64 as c64;
use num_traits::Zero;

fn main() {
    let dt = 1e-5;
    let eom = ode::GoyShell::default();
    let mut teo = semi_implicit::diag_rk4(eom, dt);
    let mut x0 = rcarr1(&vec![c64::zero(); 27]);
    x0[2] = c64::new(1.0, 0.0);
    x0[3] = c64::new(1.0, 0.0);
    x0[4] = c64::new(1.0, 0.0);
    x0[5] = c64::new(1.0, 0.0);
    x0[6] = c64::new(1.0, 0.0);
    let ts = time_series(x0, &mut teo);
    let end_time = 10_000_000;
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
            print!(",{:e},{:e}", c.re, c.im);
        }
        println!("");
    }
}
