
use fftw::*;
use ndarray::*;
use std::f64::consts::PI;

use traits::*;

pub struct KSE {
    n: usize,
    n_coef: usize,
    k: Array1<c64>,
}

pub struct KSEBuffer {
    u_pair: Pair<f64, c64>,
    ux_pair: Pair<f64, c64>,
}

impl KSEBuffer {
    /// Convert the coefficient to the field value for easy visualization
    pub fn convert_u(&mut self, u_coef: &[c64]) -> &[f64] {
        for (c, uc) in self.u_pair.coef.iter_mut().zip(u_coef.iter()) {
            *c = *uc;
        }
        self.u_pair.backward();
        &self.u_pair.field
    }
}

impl ModelSpec for KSE {
    type Scalar = c64;
    type Dim = Ix1;
    fn model_size(&self) -> usize {
        self.n_coef
    }
}

impl BufferSpec for KSE {
    type Buffer = KSEBuffer;
    fn new_buffer(&self) -> Self::Buffer {
        let u_pair = Pair::r2c_1d(self.n, FLAG::FFTW_ESTIMATE);
        let ux_pair = Pair::r2c_1d(self.n, FLAG::FFTW_ESTIMATE);
        KSEBuffer { u_pair, ux_pair }
    }
}

impl KSE {
    pub fn new(n: usize, length: f64) -> Self {
        let n_coef = n / 2 + 1;
        KSE {
            n,
            n_coef,
            k: Array::from_iter((0..n_coef).map(|i| c64::new(0.0, 2.0 * PI * i as f64 / length))),
        }
    }
}

impl SemiImplicitBuf for KSE {
    fn nlin<'a, S>(&self,
                   u: &'a mut ArrayBase<S, Self::Dim>,
                   buf: &mut Self::Buffer)
                   -> &'a mut ArrayBase<S, Self::Dim>
        where S: DataMut<Elem = Self::Scalar>
    {
        for i in 0..self.n_coef {
            buf.u_pair.coef[i] = u[i];
            buf.ux_pair.coef[i] = self.k[i] * u[i];
        }
        buf.u_pair.backward();
        buf.ux_pair.backward();
        for (up, uxp) in buf.u_pair.field.iter_mut().zip(buf.ux_pair.field.iter()) {
            *up = -*up * *uxp;
        }
        buf.u_pair.forward();
        for (up, u) in buf.u_pair.coef.iter().zip(u.iter_mut()) {
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
