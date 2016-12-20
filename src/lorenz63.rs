
use ndarray::prelude::*;

#[derive(Clone,Copy,Debug)]
pub struct Parameter {
    pub p: f64,
    pub r: f64,
    pub b: f64,
}

impl Default for Parameter {
    fn default() -> Self {
        Parameter {
            p: 10.0,
            r: 28.0,
            b: 8.0 / 3.0,
        }
    }
}

impl Parameter {
    pub fn new(p: f64, r: f64, b: f64) -> Self {
        Parameter { p: p, r: r, b: b }
    }
}

#[inline(always)]
pub fn f(p: Parameter, mut v: Array<f64, Ix1>) -> Array<f64, Ix1> {
    let x = v[0];
    let y = v[1];
    let z = v[2];
    v[0] = p.p * (y - x);
    v[1] = x * (p.r - z) - y;
    v[2] = x * y - p.b * z;
    v
}
