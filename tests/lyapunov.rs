
extern crate rand_extra;
extern crate ndarray;
extern crate ndarray_rand;
extern crate ndarray_odeint;
#[macro_use]
extern crate ndarray_linalg;

use rand_extra::*;
use ndarray::*;
use ndarray_rand::*;
use ndarray_linalg::*;
use ndarray_odeint::*;

#[test]
fn jacobian_linearity() {
    let eom = model::Lorenz63::default();
    let teo = explicit::rk4(eom, 0.01);
    let x0 = arr1(&[1.0, 0.0, 0.0]);
    let j = jacobian(&teo, x0, 1e-7);
    let dist = RealNormal::new(1.0, 1.0);
    let v = Array::random(3, dist);
    let w = Array::random(3, dist);
    let jv_jw = j.op(&v) + j.op(&w);
    let j_vw = j.op_into(v + w);
    assert_close_l2!(&jv_jw, &j_vw, 1e-5);
}
