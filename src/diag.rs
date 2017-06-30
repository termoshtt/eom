//! Solve linear diagonal dynamics

use ndarray::*;
use ndarray_linalg::Scalar;
use super::traits::*;

/// Linear ODE with diagonalized matrix (exactly solvable)
pub struct Diagonal<A, D>
    where A: Scalar,
          D: Dimension
{
    diag: RcArray<A, D>,
    diag_of_matrix: RcArray<A, D>,
    dt: A::Real,
}

impl<A, D> TimeStep<A::Real> for Diagonal<A, D>
    where A: Scalar,
          D: Dimension
{
    fn get_dt(&self) -> A::Real {
        self.dt
    }
    fn set_dt(&mut self, dt: A::Real) {
        Zip::from(&mut self.diag)
            .and(&self.diag_of_matrix)
            .apply(|a, &b| { *a = b.mul_real(dt).exp(); });
    }
}

impl<A, D> Diagonal<A, D>
    where A: Scalar,
          D: Dimension
{
    pub fn new(diag_of_matrix: RcArray<A, D>, dt: A::Real) -> Self {
        let mut diag = diag_of_matrix.clone();
        for v in diag.iter_mut() {
            *v = v.mul_real(dt).exp();
        }
        Diagonal {
            diag: diag,
            diag_of_matrix: diag_of_matrix,
            dt: dt,
        }
    }
}

impl<'a, A, S, D> TimeEvolution<A, S, D> for &'a Diagonal<A, D>
    where A: Scalar,
          S: DataMut<Elem = A>,
          D: Dimension
{
    fn iterate(self, mut x: ArrayBase<S, D>) -> ArrayBase<S, D> {
        for (val, d) in x.iter_mut().zip(self.diag.iter()) {
            *val = *val * *d;
        }
        x
    }
}
