use ndarray::*;

use crate::traits::*;

#[cfg_attr(doc, katexit::katexit)]
/// Lorenz three-variables system, "The chaotic attractor"
///
/// $$
/// \begin{align*}
///   \frac{dx}{dt} &= r(y-x) \\\\
///   \frac{dy}{dt} &= x(p-z) -y \\\\
///   \frac{dz}{dt} &= xy - bz
/// \end{align*}
/// $$
/// $(p, r, b)= (10, 28, 8/3)$ is original and commonly used parameter.
///
/// Links
/// ------
/// - Wikipedia <https://en.wikipedia.org/wiki/Lorenz_system>
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
        Lorenz63 { p, r, b }
    }
}

impl ModelSpec for Lorenz63 {
    type Scalar = f64;
    type Dim = Ix1;

    fn model_size(&self) -> usize {
        3
    }
}

impl Explicit for Lorenz63 {
    fn rhs<'a, S>(&mut self, v: &'a mut ArrayBase<S, Ix1>) -> &'a mut ArrayBase<S, Ix1>
    where
        S: DataMut<Elem = f64>,
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
    fn nlin<'a, S>(&mut self, v: &'a mut ArrayBase<S, Ix1>) -> &'a mut ArrayBase<S, Ix1>
    where
        S: DataMut<Elem = f64>,
    {
        let x = v[0];
        let y = v[1];
        let z = v[2];
        v[0] = self.p * y;
        v[1] = x * (self.r - z);
        v[2] = x * y;
        v
    }

    fn diag(&self) -> Array<f64, Ix1> {
        Array::from(vec![-self.p, -1.0, -self.b])
    }
}
