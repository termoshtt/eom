//! Define semi-implicit schemes

use ndarray::*;
use ndarray_linalg::*;

use super::traits::*;
use super::diag::{Diagonal, StiffDiagonal, diagonal};

pub struct DiagRK4<NLin, Lin, Time: RealScalar> {
    nlin: NLin,
    lin: Lin,
    dt: Time,
}

pub fn diag_rk4<A, D, EOM>(eom: EOM, dt: A::Real) -> DiagRK4<EOM, Diagonal<A, D>, A::Real>
    where A: Scalar,
          D: Dimension,
          EOM: StiffDiagonal<A, D>
{
    let diag = diagonal(&eom, dt / into_scalar(2.0));
    DiagRK4 {
        nlin: eom,
        lin: diag,
        dt: dt,
    }
}

impl<NLin, Lin, Time> TimeStep for DiagRK4<NLin, Lin, Time>
    where Time: RealScalar,
          Lin: TimeStep<Time = Time>
{
    type Time = Time;

    fn get_dt(&self) -> Self::Time {
        self.dt
    }

    fn set_dt(&mut self, dt: Self::Time) {
        self.lin.set_dt(dt / into_scalar(2.0));
    }
}

impl<D, NLin, Lin, Time> ModelSize<D> for DiagRK4<NLin, Lin, Time>
    where D: Dimension,
          NLin: ModelSize<D>,
          Lin: ModelSize<D>,
          Time: RealScalar
{
    fn model_size(&self) -> D::Pattern {
        self.nlin.model_size() // TODO check
    }
}

impl<NLin, Lin, Time> WithBuffer for DiagRK4<NLin, Lin, Time>
    where Lin: WithBuffer,
          Time: RealScalar
{
    type Buffer = Lin::Buffer;

    fn new_buffer(&self) -> Self::Buffer {
        self.lin.new_buffer()
    }
}

impl<A, D, NLin, Lin> TimeEvolution<D> for DiagRK4<NLin, Lin, A::Real>
    where A: Scalar,
          D: Dimension,
          NLin: SemiImplicit<D, Scalar = A, Time = A::Real>,
          Lin: TimeEvolution<D, Scalar = A> + TimeStep<Time = A::Real>
{
    type Scalar = A;

    fn iterate<'a, S>(&self,
                      x: &'a mut ArrayBase<S, D>,
                      mut buf: &mut Self::Buffer)
                      -> &'a mut ArrayBase<S, D>
        where S: DataMut<Elem = A>
    {
        // constants
        let dt = self.dt;
        let dt_2 = self.dt / into_scalar(2.0);
        let dt_3 = self.dt / into_scalar(3.0);
        let dt_6 = self.dt / into_scalar(6.0);
        // operators
        let l = &self.lin;
        let f = &self.nlin;
        // calc
        let mut x_ = x.to_owned();
        let mut lx = x.to_owned();
        l.iterate(&mut lx, &mut buf);
        let mut k1 = f.nlin(x);
        let k1_ = k1.to_owned();
        Zip::from(&mut *k1)
            .and(&x_)
            .apply(|k1, &x_| { *k1 = x_ + k1.mul_real(dt_2); });
        let mut k2 = f.nlin(l.iterate(k1, &mut buf));
        let k2_ = k2.to_owned();
        Zip::from(&mut *k2)
            .and(&lx)
            .apply(|k2, &lx| { *k2 = lx + k2.mul_real(dt_2); });
        let mut k3 = f.nlin(k2);
        let k3_ = k3.to_owned();
        Zip::from(&mut *k3)
            .and(&lx)
            .apply(|k3, &lx| { *k3 = lx + k3.mul_real(dt); });
        let mut k4 = f.nlin(l.iterate(k3, &mut buf));
        azip!(mut x_, k1_ in { *x_ = *x_ + k1_.mul_real(dt_6) });
        l.iterate(&mut x_, &mut buf);
        azip!(mut x_, k2_, k3_ in { *x_ = *x_ + (k2_ + k3_).mul_real(dt_3) });
        l.iterate(&mut x_, &mut buf);
        Zip::from(&mut *k4)
            .and(&x_)
            .apply(|k4, &x_| { *k4 = x_ + k4.mul_real(dt_6); });
        k4
    }
}
