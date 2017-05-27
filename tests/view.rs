
extern crate ndarray;
extern crate ndarray_odeint;

use ndarray::*;
use ndarray_odeint::*;

#[test]
fn arr() {
    let dt = 0.01;
    let eom = model::Lorenz63::default();
    let teo = explicit::euler(eom, dt);
    let x = arr1(&[1.0, 0.0, 0.0]);
    let _x = teo.iterate(x);
}

#[test]
fn rcarr() {
    let dt = 0.01;
    let eom = model::Lorenz63::default();
    let teo = explicit::euler(eom, dt);
    let x = rcarr1(&[1.0, 0.0, 0.0]);
    let _x = teo.iterate(x);
}

#[test]
fn view_mut() {
    let dt = 0.01;
    let eom = model::Lorenz63::default();
    let teo = explicit::euler(eom, dt);
    let mut x = arr1(&[1.0, 0.0, 0.0]);
    let _x = teo.iterate(x.view_mut());
}
