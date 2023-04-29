//! Kuramoto-Sivashinsky equation (KSE)
//!
//! KSE is a representative example of spatio-temporal chaos, or phase-turbulence.
//! See also the book ["Chemical Oscillations, Waves, and Turbulence"](http://www.springer.com/us/book/9783642696916), or related articles.

use fftw::types::c64;
use ndarray::*;
use std::f64::consts::PI;

use super::Pair;
use crate::traits::*;

/// One-dimensional Kuramoto-Sivashinsky equation with spectral-method
pub struct KSE {
    n: usize,
    nf: usize,
    length: f64,
    k: Array1<c64>,
    u: Pair,
    ux: Pair,
}

impl Clone for KSE {
    fn clone(&self) -> Self {
        Self::new(self.n, self.length)
    }
}

impl ModelSpec for KSE {
    type Scalar = c64;
    type Dim = Ix1;
    fn model_size(&self) -> usize {
        self.nf
    }
}

impl KSE {
    pub fn new(n: usize, length: f64) -> Self {
        let nf = n / 2 + 1;
        let k0 = 2.0 * PI / length;
        KSE {
            n,
            nf,
            length,
            k: Array::from_iter((0..nf).map(|i| c64::new(0.0, k0 * i as f64))),
            u: Pair::new(n),
            ux: Pair::new(n),
        }
    }
}

impl SemiImplicit for KSE {
    fn nlin<'a, S>(
        &mut self,
        uf: &'a mut ArrayBase<S, Self::Dim>,
    ) -> &'a mut ArrayBase<S, Self::Dim>
    where
        S: DataMut<Elem = Self::Scalar>,
    {
        azip!((u in &mut self.u.coeff_view_mut(), ux in &mut self.ux.coeff_view_mut(), &k in &self.k, &uf in &*uf) {
            *u = uf;
            *ux = k * uf;
        });
        self.u.c2r();
        self.ux.c2r();
        azip!((u in &mut self.u.real_view_mut(), &ux in &self.ux.real_view()) {
            *u = -*u * ux;
        });
        self.u.r2c();
        uf.as_slice_mut().unwrap().copy_from_slice(&self.u.c);
        uf
    }

    fn diag(&self) -> Array1<c64> {
        let k2 = &self.k * &self.k;
        let k4 = &k2 * &k2;
        -k2 - k4
    }
}
