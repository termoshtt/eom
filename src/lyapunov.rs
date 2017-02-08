//! Jacobi matrix for time-evolution function

use ndarray::*;
use ndarray_linalg::prelude::*;

use super::traits::TimeEvolution;

pub use ndarray::linalg::Dot;

pub struct Jacobian<'a, TEO>
    where TEO: 'a + TimeEvolution<Ix1>
{
    f: &'a TEO,
    x: RcArray1<f64>,
    fx: RcArray1<f64>,
    alpha: f64,
}

pub trait NumDifferentiable: Sized + TimeEvolution<Ix1> {
    fn jacobian<'a>(&'a self, x: RcArray1<f64>, alpha: f64) -> Jacobian<'a, Self>;
}

impl<TEO> NumDifferentiable for TEO
    where TEO: TimeEvolution<Ix1>
{
    fn jacobian<'a>(&'a self, x: RcArray1<f64>, alpha: f64) -> Jacobian<'a, Self> {
        let fx = self.iterate(x.clone());
        Jacobian {
            f: self,
            x: x,
            fx: fx,
            alpha: alpha,
        }
    }
}

impl<'a, S, TEO> Dot<ArrayBase<S, Ix1>> for Jacobian<'a, TEO>
    where TEO: 'a + TimeEvolution<Ix1>,
          S: Data<Elem = f64>
{
    type Output = RcArray1<f64>;
    fn dot(&self, dx: &ArrayBase<S, Ix1>) -> Self::Output {
        let nrm = self.x.norm_l2().max(dx.norm_l2());
        let n = self.alpha / nrm;
        let x = n * dx + &self.x;
        (self.f.iterate(x.into_shared()) - &self.fx) / n
    }
}

impl<'a, S, TEO> Dot<ArrayBase<S, Ix2>> for Jacobian<'a, TEO>
    where TEO: 'a + TimeEvolution<Ix1>,
          S: Data<Elem = f64>
{
    type Output = Array2<f64>;
    fn dot(&self, dxs: &ArrayBase<S, Ix2>) -> Self::Output {
        hstack(&dxs.axis_iter(Axis(1))
                .map(|dx| self.dot(&dx))
                .collect::<Vec<_>>())
            .unwrap()
    }
}
