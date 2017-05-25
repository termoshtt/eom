//! Lorenz three-variables system
//! https://en.wikipedia.org/wiki/Lorenz_system

use ndarray::*;
use super::traits::EOM;

#[derive(Clone,Copy,Debug)]
pub struct Lorenz63 {
    pub p: f64,
    pub r: f64,
    pub b: f64,
}

impl Default for Lorenz63 {
    fn default() -> Self {
        Lorenz63 {
            p: 10.0,
            r: 28.0,
            b: 8.0 / 3.0,
        }
    }
}

impl Lorenz63 {
    pub fn new(p: f64, r: f64, b: f64) -> Self {
        Lorenz63 { p: p, r: r, b: b }
    }
}

impl<'a> EOM<f64, OwnedRcRepr<f64>, Ix1> for &'a Lorenz63 {
    fn rhs(self, mut v: RcArray1<f64>) -> RcArray1<f64> {
        let x = v[0];
        let y = v[1];
        let z = v[2];
        v[0] = self.p * (y - x);
        v[1] = x * (self.r - z) - y;
        v[2] = x * y - self.b * z;
        v
    }
}

impl<'a> EOM<f64, ViewRepr<&'a mut f64>, Ix1> for &'a Lorenz63 {
    fn rhs(self, mut v: ArrayViewMut1<f64>) -> ArrayViewMut1<f64> {
        let x = v[0];
        let y = v[1];
        let z = v[2];
        v[0] = self.p * (y - x);
        v[1] = x * (self.r - z) - y;
        v[2] = x * y - self.b * z;
        v
    }
}
