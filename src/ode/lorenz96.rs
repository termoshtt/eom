use crate::traits::*;
use ndarray::*;

#[cfg_attr(doc, katexit::katexit)]
/// Lorenz 96 model, frequently used on [data assimilation](https://en.wikipedia.org/wiki/Data_assimilation) community.
///
/// $$
/// \frac{dx_i}{dt} = (x_{i+1}-x_{i-2}) x_{i-1} - x_{i} + F
/// $$
/// where $i \in [0, n-1]$ and cyclic boundary condition, i.e. $x_{n + i} = x_i$ is used.
/// $F = 8$ is used commonly to produce chaotic behavior.
///
/// Links
/// ------
/// - Wikipedia <https://en.wikipedia.org/wiki/Lorenz_96_model>
#[derive(Clone, Copy, Debug)]
pub struct Lorenz96 {
    /// $F$ in the equation, default is $8.0$
    pub f: f64,
    /// Number of elements, default is $40$
    pub n: usize,
}

impl Default for Lorenz96 {
    fn default() -> Self {
        Lorenz96 { f: 8.0, n: 40 }
    }
}

impl ModelSpec for Lorenz96 {
    type Scalar = f64;
    type Dim = Ix1;

    fn model_size(&self) -> usize {
        self.n
    }
}

impl Explicit for Lorenz96 {
    fn rhs<'a, S>(&mut self, v: &'a mut ArrayBase<S, Ix1>) -> &'a mut ArrayBase<S, Ix1>
    where
        S: DataMut<Elem = f64>,
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
