//! Solve linear diagonal dynamics

use ndarray::*;
use ndarray_linalg::Scalar;
use super::traits::*;

/// Linear ODE with diagonalized matrix (exactly solvable)
pub struct Diagonal<A, D>
    where A: Scalar,
          D: Dimension
{
    diag: Array<A, D>,
    diag_of_matrix: Array<A, D>,
    dt: A::Real,
}

impl<A, D> TimeStep for Diagonal<A, D>
    where A: Scalar,
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

impl<A, D> ModelSize<D> for Diagonal<A, D>
    where A: Scalar,
          D: Dimension
{
    fn model_size(&self) -> D::Pattern {
        self.diag.dim()
    }
}

impl<A, D> Diagonal<A, D>
    where A: Scalar,
          D: Dimension
{
    pub fn new<S: Data<Elem = A>>(diag_of_matrix: &ArrayBase<S, D>, dt: A::Real) -> Self {
        let mut diag = diag_of_matrix.to_owned();
        for v in diag.iter_mut() {
            *v = v.mul_real(dt).exp();
        }
        Diagonal {
            diag: diag,
            diag_of_matrix: diag_of_matrix.to_owned(),
            dt: dt,
        }
    }
}

impl<A, Sr, D> SemiImplicitLinear<Sr, D> for Diagonal<A, D>
    where A: Scalar,
          Sr: DataMut<Elem = A>,
          D: Dimension
{
    type Scalar = A;

    fn lin<'a>(&self, mut x: &'a mut ArrayBase<Sr, D>) -> &'a mut ArrayBase<Sr, D> {
        for (val, d) in x.iter_mut().zip(self.diag.iter()) {
            *val = *val * *d;
        }
        x
    }
}
