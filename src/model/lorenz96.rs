//! Lorenz 96 model
//! https://en.wikipedia.org/wiki/Lorenz_96_model

use ndarray::*;
use traits::Explicit;

#[derive(Clone, Copy, Debug)]
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

impl<S> Explicit<S, Ix1> for Lorenz96
    where S: DataMut<Elem = f64>
{
    type Time = f64;

    #[inline(always)]
    fn rhs<'a>(&self, mut v: &'a mut ArrayBase<S, Ix1>) -> &'a mut ArrayBase<S, Ix1> {
        let n = v.len();
        let v0 = v.to_owned();
        for i in 0..n {
            let p1 = (i + 1) % n;
            let m1 = (i + n - 1) % n;
            let m2 = (i + n - 2) % n;
            v[i] = (v0[p1] - v0[m2]) * v0[m1] - v0[i] + self.f;
        }
        v
    }
}
