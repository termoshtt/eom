use fftw::array::*;
use fftw::plan::*;
use fftw::types::*;
use ndarray::*;
use std::f64::consts::PI;

use traits::*;

/// Pair of one-dimensional Real/Complex aligned arrays
pub struct Pair {
    pub r: AlignedVec<f64>,
    pub c: AlignedVec<c64>,
    r2c: R2CPlan64,
    c2r: C2RPlan64,
}

impl Pair {
    pub fn new(n: usize) -> Self {
        let nf = n / 2 + 1;
        let mut r = AlignedVec::new(n);
        let mut c = AlignedVec::new(nf);
        let r2c = R2CPlan::new(&[n], &mut r, &mut c, Flag::Measure).unwrap();
        let c2r = C2RPlan::new(&[n], &mut c, &mut r, Flag::Measure).unwrap();
        Pair { r, c, r2c, c2r }
    }

    pub fn r2c(&mut self) {
        self.r2c.r2c(&mut self.r, &mut self.c).unwrap();
    }

    pub fn c2r(&mut self) {
        self.c2r.c2r(&mut self.c, &mut self.r).unwrap();
    }
}

impl Clone for Pair {
    fn clone(&self) -> Self {
        Pair::new(self.r.len())
    }
}

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
        for i in 0..self.nf {
            self.u.c[i] = uf[i];
            self.ux.c[i] = self.k[i] * uf[i];
        }
        self.u.c2r();
        self.ux.c2r();
        for (u, ux) in self.u.r.iter_mut().zip(self.ux.r.iter()) {
            *u = -*u * *ux;
        }
        self.u.r2c();
        let normalize = 1.0 / self.n as f64;
        for (u_, u) in uf.iter_mut().zip(self.u.c.iter()) {
            *u_ = *u * normalize;
        }
        uf
    }

    fn diag(&self) -> Array1<c64> {
        let k2 = &self.k * &self.k;
        let k4 = &k2 * &k2;
        -k2 - k4
    }
}
