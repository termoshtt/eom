//! Define semi-implicit schemes

use super::traits::*;
use super::diag::Diagonal;

use ndarray::*;
use ndarray_linalg::{Scalar, into_scalar};

pub struct DiagRK4<A, F, D>
    where A: Scalar,
          D: Dimension
{
    f: F,
    lin_half: Diagonal<A, D>,
    dt: A::Real,
}

pub fn diag_rk4<A, F, D>(f: F, dt: A::Real) -> DiagRK4<A, F, D>
    where A: Scalar,
          F: Diag<A, D>,
          D: Dimension
{
    DiagRK4::new(f, dt)
}

impl<A, F, D> DiagRK4<A, F, D>
    where A: Scalar,
          D: Dimension
{
    pub fn new(f: F, dt: A::Real) -> Self
        where F: Diag<A, D>
    {
        let diag = f.diagonal();
        let lin_half = Diagonal::new(diag, dt / into_scalar(2.0));
        DiagRK4 {
            f: f,
            lin_half: lin_half,
            dt: dt,
        }
    }
}

macro_rules! impl_time_evolution {
    ( $($mut_:tt), * ) => {

impl<'a, A, S, F, D> TimeEvolution<A, S, D> for &'a $($mut_),* DiagRK4<A, F, D>
    where A: Scalar,
          S: DataMut<Elem=A> + DataClone<Elem=A>,
          for<'b> &'b $($mut_),* F: NonLinear<A, S, D>,
          D: Dimension
{
    fn iterate(self, x: &mut ArrayBase<S, D>) -> &mut ArrayBase<S, D> {
        // constants
        let dt = self.dt;
        let dt_2 = self.dt / into_scalar(2.0);
        let dt_3 = self.dt / into_scalar(3.0);
        let dt_6 = self.dt / into_scalar(6.0);
        // operators
        let l = &self.lin_half;
        let f = &$($mut_),* self.f;
        // calc
        let mut x_ = x.to_owned();
        let mut lx = x.to_owned();
        l.iterate(&mut lx);
        let mut k1 = f.nlin(x);
        let k1_ = k1.to_owned();
        Zip::from(&mut *k1).and(&x_).apply(|k1, &x_| {
            *k1 = x_ + k1.mul_real(dt_2);
        });
        let mut k2 = f.nlin(l.iterate(k1));
        let k2_ = k2.to_owned();
        Zip::from(&mut *k2).and(&lx).apply(|k2, &lx| {
            *k2 = lx + k2.mul_real(dt_2);
        });
        let mut k3 = f.nlin(k2);
        let k3_ = k3.to_owned();
        Zip::from(&mut *k3).and(&lx).apply(|k3, &lx| {
            *k3 = lx + k3.mul_real(dt);
        });
        let mut k4 = f.nlin(l.iterate(k3));
        azip!(mut x_, k1_ in { *x_ = *x_ + k1_.mul_real(dt_6) });
        l.iterate(&mut x_);
        azip!(mut x_, k2_, k3_ in { *x_ = *x_ + (k2_ + k3_).mul_real(dt_3) });
        l.iterate(&mut x_);
        Zip::from(&mut *k4).and(&x_).apply(|k4, &x_| {
            *k4 = x_ + k4.mul_real(dt_6);
        });
        k4
    }
}

}} // impl_time_evolution!

impl_time_evolution!();
impl_time_evolution!(mut);
