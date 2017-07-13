//! Define semi-implicit schemes

use ndarray::*;
use ndarray_linalg::*;

use super::traits::*;
use super::diag::{Diagonal, StiffDiagonal, diagonal};

pub struct StiffRK4<NLin, Lin, Time: RealScalar> {
    nlin: NLin,
    lin: Lin,
    dt: Time,
}

pub fn stiff_rk4<A, D, EOM>(eom: EOM, dt: A::Real) -> StiffRK4<EOM, Diagonal<A, D>, A::Real>
    where A: Scalar,
          D: Dimension,
          EOM: StiffDiagonal<A, D>
{
    let diag = diagonal(&eom, dt / into_scalar(2.0));
    StiffRK4 {
        nlin: eom,
        lin: diag,
        dt: dt,
    }
}

impl<D, NLin, Lin, Time> ModelSize<D> for StiffRK4<NLin, Lin, Time>
    where D: Dimension,
          NLin: ModelSize<D>,
          Lin: ModelSize<D>,
          Time: RealScalar
{
    fn model_size(&self) -> D::Pattern {
        self.nlin.model_size() // TODO check
    }
}

impl<A, S, D, NLin, Lin> TimeEvolutionBase<S, D> for StiffRK4<NLin, Lin, A::Real>
    where A: Scalar,
          S: DataMut<Elem = A> + DataOwned,
          D: Dimension,
          NLin: SemiImplicitNonLinear<S, D, Scalar = A, Time = A::Real>,
          Lin: SemiImplicitLinear<S, D, Scalar = A, Time = A::Real>
{
    type Scalar = A;
    type Time = A::Real;

    fn iterate<'a>(&self, x: &'a mut ArrayBase<S, D>) -> &'a mut ArrayBase<S, D> {
        // constants
        let dt = self.dt;
        let dt_2 = self.dt / into_scalar(2.0);
        let dt_3 = self.dt / into_scalar(3.0);
        let dt_6 = self.dt / into_scalar(6.0);
        // operators
        let l = &self.lin;
        let f = &self.nlin;
        // calc
        let mut x_ = replicate(&x);
        let mut lx = replicate(&x);
        l.lin(&mut lx);
        let mut k1 = f.nlin(x);
        let k1_ = k1.to_owned();
        Zip::from(&mut *k1)
            .and(&x_)
            .apply(|k1, &x_| { *k1 = x_ + k1.mul_real(dt_2); });
        let mut k2 = f.nlin(l.lin(k1));
        let k2_ = k2.to_owned();
        Zip::from(&mut *k2)
            .and(&lx)
            .apply(|k2, &lx| { *k2 = lx + k2.mul_real(dt_2); });
        let mut k3 = f.nlin(k2);
        let k3_ = k3.to_owned();
        Zip::from(&mut *k3)
            .and(&lx)
            .apply(|k3, &lx| { *k3 = lx + k3.mul_real(dt); });
        let mut k4 = f.nlin(l.lin(k3));
        azip!(mut x_, k1_ in { *x_ = *x_ + k1_.mul_real(dt_6) });
        l.lin(&mut x_);
        azip!(mut x_, k2_, k3_ in { *x_ = *x_ + (k2_ + k3_).mul_real(dt_3) });
        l.lin(&mut x_);
        Zip::from(&mut *k4)
            .and(&x_)
            .apply(|k4, &x_| { *k4 = x_ + k4.mul_real(dt_6); });
        k4
    }
}
