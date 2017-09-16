
use ndarray::*;
use ndarray_linalg::*;
use num_traits::Float;

use super::traits::*;

/// Jacobian operator using numerical-differentiation
pub struct Jacobian<'a, S, TEO>
    where S: Data<Elem = TEO::Scalar>,
          TEO: 'a + TimeEvolution
{
    f: &'a TEO,
    x: ArrayBase<S, TEO::Dim>,
    fx: ArrayBase<S, TEO::Dim>,
    alpha: <TEO::Scalar as AssociatedReal>::Real,
}

pub fn jacobian<'a, S, TEO>(f: &'a TEO,
                            x: ArrayBase<S, TEO::Dim>,
                            alpha: <TEO::Scalar as AssociatedReal>::Real)
                            -> Jacobian<'a, S, TEO>
    where S: DataClone<Elem = TEO::Scalar> + DataMut,
          TEO: TimeEvolution
{
    let mut fx = x.clone();
    let mut buf = f.new_buffer();
    f.iterate(&mut fx, &mut buf);
    Jacobian { f, x, fx, alpha }
}

impl<'j, A, S, Sr, TEO> OperatorMut<Sr, TEO::Dim> for Jacobian<'j, S, TEO>
    where A: Scalar,
          S: Data<Elem = A>,
          Sr: DataMut<Elem = A>,
          TEO: TimeEvolution<Scalar = A>
{
    fn op_mut<'a>(&self, dx: &'a mut ArrayBase<Sr, TEO::Dim>) -> &'a mut ArrayBase<Sr, TEO::Dim> {
        let dx_nrm = dx.norm_l2().max(self.alpha);
        let n = self.alpha / dx_nrm;
        Zip::from(&mut *dx)
            .and(&self.x)
            .apply(|dx, &x| { *dx = x + dx.mul_real(n); });
        let mut buf = self.f.new_buffer();
        let x_dx = self.f.iterate(dx, &mut buf);
        Zip::from(&mut *x_dx)
            .and(&self.fx)
            .apply(|x_dx, &fx| { *x_dx = (*x_dx - fx).div_real(n); });
        x_dx
    }
}

impl<'j, A, D, S, Sr, TEO> OperatorInto<Sr, D> for Jacobian<'j, S, TEO>
    where A: Scalar,
          D: Dimension,
          S: Data<Elem = A>,
          Sr: DataMut<Elem = A>,
          TEO: TimeEvolution<Scalar = A, Dim = D>
{
    fn op_into(&self, mut dx: ArrayBase<Sr, D>) -> ArrayBase<Sr, D> {
        self.op_mut(&mut dx);
        dx
    }
}

impl<'j, A, Si, S, D, TEO> Operator<A, Si, D> for Jacobian<'j, S, TEO>
    where A: Scalar,
          S: Data<Elem = A>,
          Si: Data<Elem = A>,
          D: Dimension,
          TEO: TimeEvolution<Scalar = A, Dim = D>
{
    fn op(&self, dx: &ArrayBase<Si, D>) -> Array<A, D> {
        let dx = replicate(dx);
        self.op_into(dx)
    }
}
