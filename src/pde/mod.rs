//! Example nonlinear PDEs

pub mod kse;
pub use self::kse::KSE;

use fftw::array::*;
use fftw::plan::*;
use fftw::types::*;

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
        let n = 1.0 / self.r.len() as f64;
        for v in self.c.iter_mut() {
            *v *= n;
        }
    }

    pub fn c2r(&mut self) {
        self.c2r.c2r(&mut self.c, &mut self.r).unwrap();
    }

    pub fn to_r<'a, 'b>(&'a mut self, c: &'b [c64]) -> &'a [f64] {
        self.c.copy_from_slice(c);
        self.c2r();
        &self.r
    }

    pub fn to_c<'a, 'b>(&'a mut self, r: &'b [f64]) -> &'a [c64] {
        self.r.copy_from_slice(r);
        self.r2c();
        &self.c
    }
}

impl Clone for Pair {
    fn clone(&self) -> Self {
        Pair::new(self.r.len())
    }
}
