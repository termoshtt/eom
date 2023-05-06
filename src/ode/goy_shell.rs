use ndarray::*;
use num_complex::Complex64 as c64;
use num_traits::{PrimInt, Zero};

use crate::traits::*;

#[cfg_attr(doc, katexit::katexit)]
/// GOY-shell model, a simple model for energy cascade in turbulence
///
/// $$
/// \left(\frac{d}{dt} + \nu k^2_n \right)u_n =
/// ik_n \left(
///     a_n u_{n+1}^* u_{n+2}^*
///     + \frac{1}{2} b_n u_{n-1}^* u_{n+1}^*
///     + \frac{1}{4} c_n u_{n-1}^* u_{n-2}^*
/// \right) + f \delta_{n,m}
/// $$
///
/// where
/// - $n \in [0, N-1]$
/// - $k_n = k_0 2^n$
/// - $u_n \in \mathbb{C}$ and $u_n^*$ means its complex conjugate.
/// - $\delta_{n, m}$ means $1$ if $n = m$ and $0$ otherwise.
/// - $a_n = 1, b_n = -\epsilon, c_n = -(1-\epsilon)$ where $\epsilon$
///   represents the energy transfer within the cascade.
///
/// This is also an example for stiff equation
/// since $k_n^2$ are exponentially large.
///
/// Links
/// -----
/// - ["Transition to chaos in a shell model of turbulence", L. Biferale et al.](https://doi.org/10.1016/0167-2789(95)90065-9)
///
#[derive(Clone, Copy, Debug)]
pub struct GoyShell {
    /// $N$, system size
    size: usize,
    /// $\nu$, the viscosity
    nu: f64,
    /// $\epsilon$, Energy transfer parameter
    e: f64,
    /// $k_0$, Base wave number
    k0: f64,
    /// Amplitude of external energy input
    f: f64,
    /// $m$, the wave number of external energy input
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

impl ModelSpec for GoyShell {
    type Scalar = c64;
    type Dim = Ix1;

    fn model_size(&self) -> usize {
        self.size
    }
}

impl SemiImplicit for GoyShell {
    fn nlin<'a, S>(&mut self, v: &'a mut ArrayBase<S, Ix1>) -> &'a mut ArrayBase<S, Ix1>
    where
        S: DataMut<Elem = c64>,
    {
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

        v[self.f_idx] += c64::new(self.f, 0.0);
        v
    }

    fn diag(&self) -> Array<c64, Ix1> {
        (0..self.size)
            .map(|n| self.nu * self.k(n) * self.k(n))
            .collect()
    }
}
