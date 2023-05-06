use fftw::types::c64;
use ndarray::*;
use std::f64::consts::PI;

use super::Pair;
use crate::traits::*;

/// One-dimensional Swift-Hohenberg equation
#[derive(Clone)]
pub struct SHE {
    nf: usize,

    /// Parameter for linear stablity
    r: f64,
    /// Length scale of instablity
    qc: f64,

    k: Array1<c64>,
    u: Pair,
}

impl ModelSpec for SHE {
    type Scalar = c64;
    type Dim = Ix1;
    fn model_size(&self) -> usize {
        self.nf
    }
}

impl SHE {
    pub fn new(n: usize, length: f64, r: f64, lc: f64) -> Self {
        let nf = n / 2 + 1;
        let k0 = 2.0 * PI / length;
        let qc = 2.0 * PI / lc;
        SHE {
            nf,
            r,
            qc,
            k: Array::from_iter((0..nf).map(|i| c64::new(0.0, k0 * i as f64))),
            u: Pair::new(n),
        }
    }
}

impl SemiImplicit for SHE {
    fn nlin<'a, S>(
        &mut self,
        uf: &'a mut ArrayBase<S, Self::Dim>,
    ) -> &'a mut ArrayBase<S, Self::Dim>
    where
        S: DataMut<Elem = Self::Scalar>,
    {
        let c2 = 1.64;
        let c3 = 1.0;

        azip!((u in &mut self.u.coeff_view_mut(), &uf in &*uf) {
            *u = uf;
        });
        self.u.c2r();
        azip!((u in &mut self.u.real_view_mut()) {
            *u = c2 * *u * *u - c3 * *u * *u * *u;
        });
        self.u.r2c();
        uf.as_slice_mut().unwrap().copy_from_slice(&self.u.c);
        uf
    }

    fn diag(&self) -> Array1<c64> {
        let r = c64::new(self.r, 0.0);
        let qc2 = c64::new(self.qc.powi(2), 0.0);
        let k2 = &self.k * &self.k;
        let d = qc2 + &k2;
        r - &d * &d
    }
}
