extern crate eom;
extern crate ndarray;

use eom::traits::*;
use eom::*;
use ndarray::arr1;

fn main() {
    let dt = 0.01;
    let eom = ode::Roessler::default();
    let mut teo = explicit::RK4::new(eom, dt);
    let ts = adaptor::time_series(arr1(&[1.0, 0.0, 0.0]), &mut teo);
    let end_time = 50000;
    println!("time,x,y,z");
    for (t, v) in ts.take(end_time).enumerate() {
        println!("{},{},{},{}", t as f64 * dt, v[0], v[1], v[2]);
    }
}
