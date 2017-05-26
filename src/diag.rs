//! Solve linear diagonal dynamics

use ndarray::*;
use super::traits::*;

/// Linear ODE with diagonalized matrix (exactly solvable)
pub struct Diagonal<A, D>
    where A: OdeScalar<f64> + Exponential,
          D: Dimension
{
    diag: RcArray<A, D>,
    diag_of_matrix: RcArray<A, D>,
    dt: f64,
}

impl<A, D> Diagonal<A, D>
    where A: OdeScalar<f64> + Exponential,
          D: Dimension
{
    pub fn new(diag_of_matrix: RcArray<A, D>, dt: f64) -> Self {
        let mut diag = diag_of_matrix.clone();
        for v in diag.iter_mut() {
            *v = (*v * dt).exp();
        }
        Diagonal {
            diag: diag,
            diag_of_matrix: diag_of_matrix,
            dt: dt,
        }
    }
}

impl<A, D> TimeStep for Diagonal<A, D>
    where A: OdeScalar<f64> + Exponential,
          D: Dimension
{
    fn get_dt(&self) -> f64 {
        self.dt
    }
    fn set_dt(&mut self, dt: f64) {
        Zip::from(&mut self.diag).and(&self.diag_of_matrix).apply(|a, &b| { *a = (b * dt).exp(); });
    }
}

impl<'a, A, D> TimeEvolution<A, OwnedRcRepr<A>, D> for &'a Diagonal<A, D>
    where A: OdeScalar<f64> + Exponential,
          D: Dimension
{
    fn iterate(self, mut x: RcArray<A, D>) -> RcArray<A, D> {
        for (val, d) in x.iter_mut().zip(self.diag.iter()) {
            *val = *val * *d;
        }
        x
    }
}

impl<'a, A, D> TimeEvolution<A, ViewRepr<&'a mut A>, D> for &'a Diagonal<A, D>
    where A: OdeScalar<f64> + Exponential,
          D: Dimension
{
    fn iterate(self, mut x: ArrayViewMut<A, D>) -> ArrayViewMut<A, D> {
        for (val, d) in x.iter_mut().zip(self.diag.iter()) {
            *val = *val * *d;
        }
        x
    }
}
