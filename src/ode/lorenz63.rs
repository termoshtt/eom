//! Lorenz three-variables system
//! https://en.wikipedia.org/wiki/Lorenz_system

use ndarray::*;

use traits::*;

#[derive(Clone, Copy, Debug)]
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

impl ModelSpec for Lorenz63 {
    type Scalar = f64;
    type Dim = Ix1;

    fn model_size(&self) -> usize {
        3
    }
}

no_buffer!(Lorenz63);

impl Explicit for Lorenz63 {
    fn rhs<'a, S>(&self, v: &'a mut ArrayBase<S, Ix1>) -> &'a mut ArrayBase<S, Ix1>
        where S: DataMut<Elem = f64>
    {
        let x = v[0];
        let y = v[1];
        let z = v[2];
        v[0] = self.p * (y - x);
        v[1] = x * (self.r - z) - y;
        v[2] = x * y - self.b * z;
        v
    }
}

impl SemiImplicit for Lorenz63 {
    fn nlin<'a, S>(&self, v: &'a mut ArrayBase<S, Ix1>) -> &'a mut ArrayBase<S, Ix1>
        where S: DataMut<Elem = f64>
    {
        let x = v[0];
        let y = v[1];
        let z = v[2];
        v[0] = self.p * y;
        v[1] = x * (self.r - z);
        v[2] = x * y;
        v
    }
}

impl StiffDiagonal for Lorenz63 {
    fn diag(&self) -> Array<f64, Ix1> {
        Array::from_vec(vec![-self.p, -1.0, -self.b])
    }
}
