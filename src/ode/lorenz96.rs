//! Lorenz 96 model
//! https://en.wikipedia.org/wiki/Lorenz_96_model

use ndarray::*;
use traits::*;

#[derive(Clone, Copy, Debug, new)]
pub struct Lorenz96 {
    pub f: f64,
    pub n: usize,
}

impl Default for Lorenz96 {
    fn default() -> Self {
        Lorenz96 { f: 8.0, n: 40 }
    }
}

no_buffer!(Lorenz96);

impl ModelSpec for Lorenz96 {
    type Dim = Ix1;

    fn model_size(&self) -> usize {
        self.n
    }
}

impl Explicit for Lorenz96 {
    type Scalar = f64;

    fn rhs<'a, S>(&self, mut v: &'a mut ArrayBase<S, Ix1>) -> &'a mut ArrayBase<S, Ix1>
        where S: DataMut<Elem = f64>
    {
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
