
extern crate rand_extra;
extern crate ndarray;
extern crate ndarray_rand;
extern crate ndarray_odeint;
#[macro_use]
extern crate ndarray_linalg;

use rand_extra::*;
use ndarray::*;
use ndarray_rand::*;
use ndarray_linalg::prelude::*;
use ndarray_odeint::*;
use ndarray_odeint::lyapunov::*;

#[test]
fn jacobian_linearity() {
    let eom = model::Lorenz63::default();
    let teo = explicit::rk4(eom, 0.01);
    let x0 = rcarr1(&[1.0, 0.0, 0.0]);
    let j = jacobian(&teo, x0, 1e-7);
    let dist = RealNormal::new(1.0, 1.0);
    let v = Array::random(3, dist);
    let w = Array::random(3, dist);
    assert_close_l2!(&(j.dot(&v) + j.dot(&w)), &j.dot(&(v + w)), 1e-5);
}

#[test]
fn jacobian_matrix_shape() {
    let eom = model::Lorenz63::default();
    let teo = explicit::rk4(eom, 0.01);
    let x0 = rcarr1(&[1.0, 0.0, 0.0]);
    let j = jacobian(&teo, x0, 1e-7);
    let dist = RealNormal::new(1.0, 1.0);
    let v = Array::random((3, 2), dist);
    let jv = j.dot(&v);
    assert!(jv.shape() == v.shape());
}

#[test]
#[ignore]
fn exponents_l63() {
    let dt = 0.01;
    let eom = model::Lorenz63::default();
    let teo = explicit::rk4(eom, dt);
    let l = exponents(&teo, rcarr1(&[1.0, 0.0, 0.0]), 1e-7, 100000);
    assert_close_l2!(&l, &arr1(&[0.906, 0.0, -14.572]), 1e-2);
    // value from http://sprott.physics.wisc.edu/chaos/lorenzle.htm
}
