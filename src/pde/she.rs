use fftw::types::c64;
use ndarray::*;
use std::f64::consts::PI;

use super::Pair;
use crate::traits::*;

#[cfg_attr(doc, katexit::katexit)]
/// One-dimensional Swift-Hohenberg equation with spectral method
///
/// $$
/// \frac{\partial u}{\partial t} =
/// ru - \left(\partial_x^2 + q_c^2 \right)^2 u + u^2 - u^3
/// $$
///
/// where $u = u(x, t)$ is real value field defined on $x \in [0, L]$
/// with cyclic boundary condition $u(x, t) = u(x + L, t)$.
/// The nonlinear term $u^2 - u^3$ has several variation,
/// and sometimes called generalized Swift-Hohenberg equation for this case.
///
/// Links
/// -----
/// - ["Localized states in the generalized Swift-Hohenberg equation", J. Burke and E. Knobloch, PRE 73, 056211 (2006)](https://journals.aps.org/pre/abstract/10.1103/PhysRevE.73.056211)
///
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
    /// - `n`: Number of Fourier coefficients to be computed
    /// - `length`: System size $L$
    /// - `r`: Stability parameter $r$
    /// - `lc`: Length scale of instablity, i.e. $q_c = 2\pi / l_c$
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
