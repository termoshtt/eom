
use ndarray::*;
use ndarray_linalg::*;
use num_traits::Float;

use super::traits::*;

/// Jacobian operator using numerical-differentiation
pub struct Jacobian<'jac, A, D, TEO>
    where A: Scalar,
          D: Dimension,
          TEO: 'jac + TimeEvolution<Scalar = A, Dim = D>
{
    f: &'jac mut TEO,
    x: Array<A, D>,
    fx: Array<A, D>,
    alpha: A::Real,
}

pub trait LinearApprox<A, D, TEO>
    where A: Scalar,
          D: Dimension,
          TEO: TimeEvolution<Scalar = A, Dim = D>
{
    fn lin_approx<'jac>(&'jac mut self,
                        x: Array<A, D>,
                        alpha: A::Real)
                        -> Jacobian<'jac, A, D, TEO>
        where TEO: 'jac;
}

impl<A, D, TEO> LinearApprox<A, D, TEO> for TEO
    where A: Scalar,
          D: Dimension,
          TEO: TimeEvolution<Scalar = A, Dim = D>
{
    fn lin_approx<'jac>(&'jac mut self, x: Array<A, D>, alpha: A::Real) -> Jacobian<'jac, A, D, TEO>
        where TEO: 'jac
    {
        Jacobian::new(self, x, alpha)
    }
}

impl<'jac, A, D, TEO> Jacobian<'jac, A, D, TEO>
    where A: Scalar,
          D: Dimension,
          TEO: TimeEvolution<Scalar = A, Dim = D>
{
    pub fn new(f: &'jac mut TEO, x: Array<A, D>, alpha: A::Real) -> Jacobian<'jac, A, D, TEO>
        where TEO: 'jac
    {
        let mut fx = x.clone();
        f.iterate(&mut fx);
        Jacobian { f, x, fx, alpha }
    }

    pub fn apply(&mut self, mut dx: Array<A, D>) -> Array<A, D> {
        self.apply_inplace(&mut dx);
        dx
    }

    pub fn apply_inplace<S>(&mut self, dx: &mut ArrayBase<S, D>)
        where S: DataMut<Elem = A>
    {
        let dx_nrm = dx.norm_l2().max(self.alpha);
        let n = self.alpha / dx_nrm;
        Zip::from(&mut *dx)
            .and(&self.x)
            .apply(|dx, &x| { *dx = x + dx.mul_real(n); });
        let x_dx = self.f.iterate(dx);
        Zip::from(&mut *x_dx)
            .and(&self.fx)
            .apply(|x_dx, &fx| { *x_dx = (*x_dx - fx).div_real(n); });
    }
}

impl<'jac, A, D, TEO> Jacobian<'jac, A, D, TEO>
    where A: Scalar,
          D: Dimension,
          TEO: TimeEvolution<Scalar = A, Dim = D>
{
    pub fn apply_multi<S>(&mut self, mut a: Array<A, D::Larger>) -> Array<A, D::Larger>
        where D::Larger: RemoveAxis + Dimension<Smaller = D>
    {
        self.apply_multi_inplace(&mut a);
        a
    }

    pub fn apply_multi_inplace<S>(&mut self, a: &mut ArrayBase<S, D::Larger>)
        where S: DataMut<Elem = A>,
              D::Larger: RemoveAxis + Dimension<Smaller = D>
    {
        let n = a.ndim();
        for mut col in a.axis_iter_mut(Axis(n - 1)) {
            self.apply_inplace(&mut col);
        }
    }
}
