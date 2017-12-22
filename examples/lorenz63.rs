
extern crate ndarray;
extern crate eom;

use ndarray::arr1;
use eom::*;

fn main() {
    let dt = 0.01;
    let eom = ode::Lorenz63::default();
    let mut teo = explicit::RK4::new(eom, dt);
    let ts = time_series(arr1(&[1.0, 0.0, 0.0]), &mut teo);
    let end_time = 10000;
    println!("time,x,y,z");
    for (t, v) in ts.take(end_time).enumerate() {
        println!("{},{},{},{}", t as f64 * dt, v[0], v[1], v[2]);
    }
}
