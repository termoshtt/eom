
extern crate ndarray;
extern crate ndarray_odeint;

use ndarray::*;
use ndarray_odeint::prelude::*;

#[test]
fn view_mut() {
    let dt = 0.01;
    let eom = Lorenz63::default();
    let teo = explicit::rk4(eom, dt);
    let mut x: Array1<f64> = arr1(&[1.0, 0.0, 0.0]);
    let _x = teo.iterate(x.view_mut());
}
