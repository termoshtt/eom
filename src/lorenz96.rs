//! Lorenz 96 model
//! https://en.wikipedia.org/wiki/Lorenz_96_model

use ndarray::prelude::*;
use super::traits::EOM;

#[derive(Clone,Copy,Debug)]
pub struct Lorenz96 {
    pub f: f64,
}

impl Default for Lorenz96 {
    fn default() -> Self {
        Lorenz96 { f: 8.0 }
    }
}

impl Lorenz96 {
    pub fn new(f: f64) -> Self {
        Lorenz96 { f: f }
    }
}

impl EOM<f64, Ix1> for Lorenz96 {
    #[inline(always)]
    fn rhs(&self, mut v: RcArray<f64, Ix1>) -> RcArray<f64, Ix1> {
        let n = v.len();
        let v0 = v.clone();
        for i in 0..n {
            let p1 = (i + 1) % n;
            let m1 = (i + n - 1) % n;
            let m2 = (i + n - 2) % n;
            v[i] = (v0[p1] - v0[m2]) * v0[m1] - v0[i] + self.f;
        }
        v
    }
}
