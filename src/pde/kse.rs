
use fftw::*;
use ndarray::*;
use std::f64::consts::PI;

use traits::*;

pub struct KSE {
    n: usize,
    n_coef: usize,
    k: Array1<c64>,
    u_pair: Pair<f64, c64>,
    ux_pair: Pair<f64, c64>,
}

impl ModelSpec for KSE {
    type Scalar = c64;
    type Dim = Ix1;
    fn model_size(&self) -> usize {
        self.n_coef
    }
}

impl KSE {
    pub fn new(n: usize, length: f64) -> Self {
        let n_coef = n / 2 + 1;
        let k = Array::from_iter((0..n_coef).map(|i| c64::new(0.0, 2.0 * PI * i as f64 / length)));
        let u_pair = Pair::r2c_1d(n, FLAG::FFTW_ESTIMATE);
        let ux_pair = Pair::r2c_1d(n, FLAG::FFTW_ESTIMATE);
        KSE {
            n,
            n_coef,
            k,
            u_pair,
            ux_pair,
        }
    }

    /// Convert the coefficient to the field value for easy visualization
    pub fn convert_u(&mut self, u_coef: &[c64]) -> &[f64] {
        for (c, uc) in self.u_pair.coef.iter_mut().zip(u_coef.iter()) {
            *c = *uc;
        }
        self.u_pair.backward();
        &self.u_pair.field
    }
}

impl SemiImplicit for KSE {
    fn nlin<'a, S>(&mut self, u: &'a mut ArrayBase<S, Self::Dim>) -> &'a mut ArrayBase<S, Self::Dim>
        where S: DataMut<Elem = Self::Scalar>
    {
        for i in 0..self.n_coef {
            self.u_pair.coef[i] = u[i];
            self.ux_pair.coef[i] = self.k[i] * u[i];
        }
        self.u_pair.backward();
        self.ux_pair.backward();
        for (up, uxp) in self.u_pair.field.iter_mut().zip(self.ux_pair.field.iter()) {
            *up = -*up * *uxp;
        }
        self.u_pair.forward();
        for (up, u) in self.u_pair.coef.iter().zip(u.iter_mut()) {
            *u = *up / self.n as f64;
        }
        u
    }
}

impl StiffDiagonal for KSE {
    fn diag(&self) -> Array1<c64> {
        let k2 = &self.k * &self.k;
        let k4 = &k2 * &k2;
        -k2 - k4
    }
}
