use ndarray::*;
use ndarray_linalg::*;
use std::f64::consts::PI;
use std::iter::FromIterator;

use eom::pde::*;

#[test]
fn pair_r2c2r() {
    let n = 128;
    let a: Array1<f64> = random(n);
    let mut p = Pair::new(n);
    p.r.copy_from_slice(a.as_slice().unwrap());
    p.r2c();
    p.c2r();
    let b: Array1<f64> = Array::from_iter(p.r.iter().cloned());
    assert_close_l2!(&a, &b, 1e-7);
}

#[test]
fn pair_c2r() {
    let n = 128;
    let k0 = 2.0 * PI / n as f64;
    let a = Array::from_shape_fn(n, |i| 2.0 * (i as f64 * k0).cos());
    let mut p = Pair::new(n);
    p.c[1] = c64::new(1.0, 0.0);
    p.c2r();
    let b = Array::from_iter(p.r.iter().cloned());
    assert_close_l2!(&a, &b, 1e-7);
}
