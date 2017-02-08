
extern crate ndarray;
extern crate ndarray_odeint;
extern crate ndarray_numtest;
extern crate ndarray_linalg;

use ndarray::*;
use ndarray_linalg::prelude::*;
use ndarray_numtest::prelude::*;

use ndarray_odeint::prelude::*;
use ndarray_odeint::lyapunov::*;

#[test]
fn jacobian_linearity() {
    let eom = Lorenz63::default();
    let teo = explicit::rk4(eom, 0.01);
    let x0 = rcarr1(&[1.0, 0.0, 0.0]);
    let j = teo.jacobian(x0, 1e-7);
    let v = Array1::real_normal_init(3, 1.0, 1.0);
    let w = Array1::real_normal_init(3, 1.0, 1.0);
    all_close_l2(&(j.dot(&v) + j.dot(&w)), &j.dot(&(v + w)), 1e-5).unwrap();
}

#[test]
fn jacobian_matrix_shape() {
    let eom = Lorenz63::default();
    let teo = explicit::rk4(eom, 0.01);
    let x0 = rcarr1(&[1.0, 0.0, 0.0]);
    let j = teo.jacobian(x0, 1e-7);
    let v = Array2::real_normal_init((3, 2), 1.0, 1.0);
    let jv = j.dot(&v);
    assert!(jv.shape() == v.shape());
}

#[test]
fn exponents_l63() {
    let dt = 0.01;
    let eom = Lorenz63::default();
    let teo = explicit::rk4(eom, dt);
    let l = exponents(&teo, rcarr1(&[1.0, 0.0, 0.0]), 1e-7, 100000);
    all_close_max(&l, &arr1(&[0.906, 0.0, -14.572]), 1e-3).unwrap();
    // value from http://sprott.physics.wisc.edu/chaos/lorenzle.htm
}
