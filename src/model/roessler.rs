//! Roessler system
//! https://en.wikipedia.org/wiki/Lorenz_syste://en.wikipedia.org/wiki/R%C3%B6ssler_attractor

use ndarray::*;
use traits::*;

#[derive(Clone, Copy, Debug)]
pub struct Roessler {
    pub a: f64,
    pub b: f64,
    pub c: f64,
}

impl Default for Roessler {
    fn default() -> Self {
        Roessler {
            a: 0.2,
            b: 0.2,
            c: 5.7,
        }
    }
}

impl ModelSize<Ix1> for Roessler {
    fn model_size(&self) -> usize {
        3
    }
}

impl Roessler {
    pub fn new(a: f64, b: f64, c: f64) -> Self {
        Roessler { a: a, b: b, c: c }
    }
}

impl<S> Explicit<S, Ix1> for Roessler
    where S: DataMut<Elem = f64>
{
    type Scalar = f64;
    type Time = f64;

    fn rhs<'a>(&self, mut v: &'a mut ArrayBase<S, Ix1>) -> &'a mut ArrayBase<S, Ix1> {
        let x = v[0];
        let y = v[1];
        let z = v[2];
        v[0] = -y - z;
        v[1] = x + self.a * y;
        v[2] = self.b + x * z - self.c * z;
        v
    }
}
