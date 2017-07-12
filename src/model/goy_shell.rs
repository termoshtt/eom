//! GOY-shell model

use ndarray::*;
use num_traits::{PrimInt, Zero};
use num_complex::Complex64 as c64;

use traits::*;

#[derive(Clone, Copy, Debug, new)]
pub struct GoyShell {
    size: usize,
    nu: f64,
    e: f64,
    k0: f64,
    f: f64,
    f_idx: usize,
}

impl GoyShell {
    fn k(&self, n: usize) -> c64 {
        c64::new(0.0, self.k0 * 2.pow(n as u32) as f64)
    }
}

impl Default for GoyShell {
    fn default() -> Self {
        GoyShell {
            size: 27,
            nu: 1e-9,
            e: 0.5,
            k0: 0.0625,
            f: 5e-3,
            f_idx: 4,
        }
    }
}

impl<Sn, Sd> SemiImplicitDiag<Sn, Sd, Ix1> for GoyShell
    where Sn: DataMut<Elem = c64>,
          Sd: DataOwned<Elem = c64>
{
    type Time = f64;

    fn nlin<'a>(&self, mut v: &'a mut ArrayBase<Sn, Ix1>) -> &'a mut ArrayBase<Sn, Ix1> {
        let mut am2 = c64::zero();
        let mut am1 = c64::zero();
        let mut a_0 = v[0].conj();
        let mut ap1 = v[1].conj();
        let mut ap2 = v[2].conj();

        let a = 1.0;
        let b = -self.e;
        let c = -(1.0 - self.e);

        for i in 0..self.size {
            v[i] = self.k(i) * (a * ap1 * ap2 + 0.5 * b * ap1 * am1 + 0.25 * c * am1 * am2);
            am2 = am1;
            am1 = a_0;
            a_0 = ap1;
            ap1 = ap2;
            if i + 3 < self.size {
                ap2 = v[i + 3].conj();
            } else {
                ap2 = c64::zero();
            }
        }

        v[self.f_idx] = v[self.f_idx] + c64::new(self.f, 0.0);
        v
    }

    fn diag(&self) -> ArrayBase<Sd, Ix1> {
        (0..self.size)
            .map(|n| self.nu * self.k(n) * self.k(n))
            .collect()
    }
}
