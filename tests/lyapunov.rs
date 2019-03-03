use ndarray::*;
use ndarray_linalg::*;

use eom::lyapunov::*;
use eom::traits::*;
use eom::*;

#[test]
fn jacobian_linearity() {
    let eom = ode::Lorenz63::default();
    let mut teo = explicit::RK4::new(eom, 0.01);
    let x0 = arr1(&[1.0, 0.0, 0.0]);
    let mut j = teo.lin_approx(x0, 1e-7);
    let v: Array1<f64> = generate::random(3);
    let w: Array1<f64> = generate::random(3);
    let j_z = j.apply(&v + &w);
    let jv_jw = j.apply(v) + j.apply(w);
    assert_close_l2!(&jv_jw, &j_z, 1e-5);
}

#[test]
fn jacobian_view() {
    let eom = ode::Lorenz63::default();
    let mut teo = explicit::RK4::new(eom, 0.01);
    let x0 = arr1(&[1.0, 0.0, 0.0]);
    let mut j = teo.lin_approx(x0, 1e-7);
    let mut v: Array1<f64> = generate::random(3);
    j.apply_inplace(&mut v.view_mut());
}

#[test]
fn jacobian_2d() {
    let eom = ode::Lorenz63::default();
    let mut teo = explicit::RK4::new(eom, 0.01);
    let x0 = arr1(&[1.0, 0.0, 0.0]);
    let mut j = teo.lin_approx(x0, 1e-7);
    let v: Array2<f64> = generate::random((3, 2));
    j.apply_multi(v);
}
