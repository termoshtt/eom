
extern crate ndarray;
extern crate eom;

use ndarray::rcarr1;
use eom::*;

fn main() {
    let dt = 0.01;
    let eom = model::Roessler::default();
    let teo = explicit::rk4(eom, dt);
    let ts = time_series(rcarr1(&[1.0, 0.0, 0.0]), &teo);
    let end_time = 50000;
    println!("time,x,y,z");
    for (t, v) in ts.take(end_time).enumerate() {
        println!("{},{},{},{}", t as f64 * dt, v[0], v[1], v[2]);
    }
}
