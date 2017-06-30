
use ndarray::*;
use ndarray_linalg::*;
use num_traits::Float;

use super::traits::*;

/// Jacobian operator using numerical-differentiation
pub struct Jacobian<'a, A, S, D, TEO>
    where A: Scalar,
          S: Data<Elem = A>,
          D: Dimension,
          TEO: 'a
{
    f: &'a TEO,
    x: ArrayBase<S, D>,
    fx: ArrayBase<S, D>,
    alpha: A::Real,
}

pub fn jacobian<'a, A, S, D, TEO>(f: &'a TEO,
                                  x: ArrayBase<S, D>,
                                  alpha: A::Real)
                                  -> Jacobian<'a, A, S, D, TEO>
    where A: Scalar,
          S: DataClone<Elem = A> + DataMut,
          D: Dimension,
          for<'b> &'b TEO: TimeEvolution<S, D>
{
    let mut fx = x.clone();
    f.iterate(&mut fx);
    Jacobian {
        f: f,
        x: x,
        fx: fx,
        alpha: alpha,
    }
}

impl<'a, 'j, A, S, D, TEO> Operator<&'a mut ArrayBase<S, D>, &'a mut ArrayBase<S, D>>
    for Jacobian<'j, A, S, D, TEO>
    where A: Scalar,
          S: DataMut<Elem = A>,
          D: Dimension,
          for<'b> &'b TEO: TimeEvolution<S, D>
{
    fn op(&self, dx: &'a mut ArrayBase<S, D>) -> &'a mut ArrayBase<S, D> {
        let dx_nrm = dx.norm_l2().max(self.alpha);
        let n = self.alpha / dx_nrm;
        Zip::from(&mut *dx)
            .and(&self.x)
            .apply(|dx, &x| { *dx = x + dx.mul_real(n); });
        let x_dx = self.f.iterate(dx);
        Zip::from(&mut *x_dx)
            .and(&self.fx)
            .apply(|x_dx, &fx| { *x_dx = (*x_dx - fx).div_real(n); });
        x_dx
    }
}
