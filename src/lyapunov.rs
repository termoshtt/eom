
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

impl<'a, 'j, A, S, Sr, D, TEO> Operator<&'a mut ArrayBase<Sr, D>, &'a mut ArrayBase<Sr, D>>
    for Jacobian<'j, A, S, D, TEO>
    where A: Scalar,
          S: Data<Elem = A>,
          Sr: DataMut<Elem = A>,
          D: Dimension,
          for<'b> &'b TEO: TimeEvolution<Sr, D>
{
    fn op(&self, dx: &'a mut ArrayBase<Sr, D>) -> &'a mut ArrayBase<Sr, D> {
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

impl<'j, A, S, Sr, D, TEO> Operator<ArrayBase<Sr, D>, ArrayBase<Sr, D>>
    for Jacobian<'j, A, S, D, TEO>
    where A: Scalar,
          S: Data<Elem = A>,
          Sr: DataMut<Elem = A>,
          D: Dimension,
          for<'b> &'b TEO: TimeEvolution<Sr, D>
{
    fn op(&self, mut dx: ArrayBase<Sr, D>) -> ArrayBase<Sr, D> {
        self.op(&mut dx);
        dx
    }
}

impl<'a, 'j, A, Si, S, D, TEO> Operator<&'a ArrayBase<Si, D>, Array<A, D>>
    for Jacobian<'j, A, S, D, TEO>
    where A: Scalar,
          S: Data<Elem = A>,
          Si: Data<Elem = A>,
          D: Dimension,
          for<'b> &'b TEO: TimeEvolution<OwnedRepr<A>, D>
{
    fn op(&self, dx: &'a ArrayBase<Si, D>) -> Array<A, D> {
        let dx = replicate(dx);
        self.op(dx)
    }
}
