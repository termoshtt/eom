//! Solve linear diagonal dynamics

use ndarray::*;
use ndarray_linalg::Scalar;
use super::traits::*;

/// Linear ODE with diagonalized matrix (exactly solvable)
pub struct Diagonal<A, S, D>
    where A: Scalar,
          S: Data<Elem = A>,
          D: Dimension
{
    diag: ArrayBase<S, D>,
    diag_of_matrix: ArrayBase<S, D>,
    dt: A::Real,
}

impl<A, S, D> TimeStep for Diagonal<A, S, D>
    where A: Scalar,
          S: DataMut<Elem = A>,
          D: Dimension
{
    type Time = A::Real;

    fn get_dt(&self) -> Self::Time {
        self.dt
    }
    fn set_dt(&mut self, dt: Self::Time) {
        Zip::from(&mut self.diag)
            .and(&self.diag_of_matrix)
            .apply(|a, &b| { *a = b.mul_real(dt).exp(); });
    }
}

impl<A, S, D> ModelSize<D> for Diagonal<A, S, D>
    where A: Scalar,
          S: Data<Elem = A>,
          D: Dimension
{
    fn model_size(&self) -> D::Pattern {
        self.diag.dim()
    }
}

impl<A, S, D> Diagonal<A, S, D>
    where A: Scalar,
          S: DataClone<Elem = A> + DataMut,
          D: Dimension
{
    pub fn new(diag_of_matrix: ArrayBase<S, D>, dt: A::Real) -> Self {
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

impl<A, S, D> TimeEvolution<S, D> for Diagonal<A, S, D>
    where A: Scalar,
          S: DataMut<Elem = A>,
          D: Dimension
{
    type Time = A::Real;
    fn iterate<'a>(&self, mut x: &'a mut ArrayBase<S, D>) -> &'a mut ArrayBase<S, D> {
        for (val, d) in x.iter_mut().zip(self.diag.iter()) {
            *val = *val * *d;
        }
        x
    }
}
