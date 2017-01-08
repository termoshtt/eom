
use ndarray::prelude::*;
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

impl EOM<Ix1> for Lorenz63 {
    #[inline(always)]
    fn rhs(&self, mut state: RcArray<f64, Ix1>) -> RcArray<f64, Ix1> {
        {
            let mut v = state.view_mut();
            let x = v[0];
            let y = v[1];
            let z = v[2];
            v[0] = self.p * (y - x);
            v[1] = x * (self.r - z) - y;
            v[2] = x * y - self.b * z;
        }
        state
    }
}
