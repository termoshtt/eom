//! Define semi-implicit schemes

use super::traits::*;
use super::diag::Diagonal;

use ndarray::*;
use ndarray_linalg::{Scalar, into_scalar, replicate};

pub struct DiagRK4<A, S, F, D>
    where A: Scalar,
          S: Data<Elem = A>,
          D: Dimension
{
    f: F,
    lin_half: Diagonal<A, S, D>,
    dt: A::Real,
}

pub fn diag_rk4<A, S, F, D>(f: F, dt: A::Real) -> DiagRK4<A, S, F, D>
    where A: Scalar,
          S: DataClone<Elem = A> + DataMut,
          F: SemiImplicitDiag<S, S, D>,
          D: Dimension
{
    DiagRK4::new(f, dt)
}

impl<A, S, F, D> DiagRK4<A, S, F, D>
    where A: Scalar,
          S: DataClone<Elem = A> + DataMut,
          D: Dimension
{
    pub fn new(f: F, dt: A::Real) -> Self
        where F: SemiImplicitDiag<S, S, D>
    {
        let diag = f.diag();
        let lin_half = Diagonal::new(diag, dt / into_scalar(2.0));
        DiagRK4 {
            f: f,
            lin_half: lin_half,
            dt: dt,
        }
    }
}


impl<A, S, F, D> TimeEvolution<S, D> for DiagRK4<A, S, F, D>
    where A: Scalar,
          S: DataMut<Elem = A> + DataClone + DataOwned,
          F: SemiImplicitDiag<S, S, D, Time = A::Real>,
          D: Dimension
{
    type Time = F::Time;

    fn iterate<'a>(&self, x: &'a mut ArrayBase<S, D>) -> &'a mut ArrayBase<S, D> {
        // constants
        let dt = self.dt;
        let dt_2 = self.dt / into_scalar(2.0);
        let dt_3 = self.dt / into_scalar(3.0);
        let dt_6 = self.dt / into_scalar(6.0);
        // operators
        let l = &self.lin_half;
        let f = &self.f;
        // calc
        let mut x_ = replicate(&x);
        let mut lx = replicate(&x);
        l.iterate(&mut lx);
        let mut k1 = f.nlin(x);
        let k1_ = k1.to_owned();
        Zip::from(&mut *k1)
            .and(&x_)
            .apply(|k1, &x_| { *k1 = x_ + k1.mul_real(dt_2); });
        let mut k2 = f.nlin(l.iterate(k1));
        let k2_ = k2.to_owned();
        Zip::from(&mut *k2)
            .and(&lx)
            .apply(|k2, &lx| { *k2 = lx + k2.mul_real(dt_2); });
        let mut k3 = f.nlin(k2);
        let k3_ = k3.to_owned();
        Zip::from(&mut *k3)
            .and(&lx)
            .apply(|k3, &lx| { *k3 = lx + k3.mul_real(dt); });
        let mut k4 = f.nlin(l.iterate(k3));
        azip!(mut x_, k1_ in { *x_ = *x_ + k1_.mul_real(dt_6) });
        l.iterate(&mut x_);
        azip!(mut x_, k2_, k3_ in { *x_ = *x_ + (k2_ + k3_).mul_real(dt_3) });
        l.iterate(&mut x_);
        Zip::from(&mut *k4)
            .and(&x_)
            .apply(|k4, &x_| { *k4 = x_ + k4.mul_real(dt_6); });
        k4
    }
}
