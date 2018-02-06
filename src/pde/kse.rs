use fftw::array::*;
use fftw::plan::*;
use fftw::types::*;
use ndarray::*;
use std::f64::consts::PI;

use traits::*;

pub struct KSE {
    n: usize,
    nf: usize,
    length: f64,
    k: Array1<c64>,
    u: AlignedVec<f64>,
    ux: AlignedVec<f64>,
    uf: AlignedVec<c64>,
    uxf: AlignedVec<c64>,
    r2c: R2CPlan64,
    c2r: C2RPlan64,
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
        let k = Array::from_iter((0..nf).map(|i| c64::new(0.0, k0 * i as f64)));
        let mut u = AlignedVec::new(n);
        let mut uf = AlignedVec::new(nf);
        let ux = AlignedVec::new(n);
        let uxf = AlignedVec::new(nf);
        let r2c = R2CPlan::new(&[n], &mut u, &mut uf, Flag::Measure).unwrap();
        let c2r = C2RPlan::new(&[n], &mut uf, &mut u, Flag::Measure).unwrap();
        KSE {
            n,
            nf,
            length,
            k,
            u,
            ux,
            uf,
            uxf,
            r2c,
            c2r,
        }
    }

    /// Convert the coefficient to the field value for easy visualization
    pub fn convert_u<'a>(&'a mut self, uf: &[c64]) -> &'a [f64] {
        for (c, uc) in self.uf.iter_mut().zip(uf.iter()) {
            *c = *uc;
        }
        self.c2r.c2r(&mut self.uf, &mut self.u).unwrap();
        &self.u
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
            self.uf[i] = uf[i];
            self.uxf[i] = self.k[i] * uf[i];
        }
        self.c2r.c2r(&mut self.uf, &mut self.u).unwrap();
        self.c2r.c2r(&mut self.uxf, &mut self.ux).unwrap();
        for (u, ux) in self.u.iter_mut().zip(self.ux.iter()) {
            *u = -*u * *ux;
        }
        self.r2c.r2c(&mut self.u, &mut self.uf).unwrap();
        let normalize = 1.0 / self.n as f64;
        for (u_, u) in uf.iter_mut().zip(self.uf.iter()) {
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
