//! Define semi-implicit schemes

use ndarray::*;
use ndarray_linalg::*;

use super::traits::*;
use super::diag::{Diagonal, diagonal};

pub struct DiagRK4<NLin, Lin>
    where Lin: TimeStep
{
    nlin: NLin,
    lin: Lin,
    dt: Lin::Time,
}

pub fn diag_rk4<A, D, EOM>(eom: EOM, dt: A::Real) -> DiagRK4<EOM, Diagonal<A, D>>
    where A: Scalar,
          D: Dimension,
          EOM: StiffDiagonal<Scalar = A, Dim = D>
{
    let diag = diagonal(&eom, dt / into_scalar(2.0));
    DiagRK4 {
        nlin: eom,
        lin: diag,
        dt: dt,
    }
}

impl<NLin, Lin> TimeStep for DiagRK4<NLin, Lin>
    where Lin: TimeStep
{
    type Time = Lin::Time;

    fn get_dt(&self) -> Self::Time {
        self.dt
    }

    fn set_dt(&mut self, dt: Self::Time) {
        self.lin.set_dt(dt / into_scalar(2.0));
    }
}

impl<D, NLin, Lin> ModelSpec for DiagRK4<NLin, Lin>
    where D: Dimension,
          NLin: ModelSpec<Dim = D>,
          Lin: ModelSpec<Dim = D> + TimeStep
{
    type Dim = D;

    fn model_size(&self) -> <Self::Dim as Dimension>::Pattern {
        self.nlin.model_size() // TODO check
    }
}

pub struct DiagRK4Buffer<NLinBuf, LinBuf, Arr> {
    nlin: NLinBuf,
    lin: LinBuf,
    x: Arr,
    lx: Arr,
    k1: Arr,
    k2: Arr,
    k3: Arr,
}

impl<A, D, NLin, Lin> BufferSpec for DiagRK4<NLin, Lin>
    where A: Scalar,
          D: Dimension,
          NLin: SemiImplicitBuf<Scalar = A, Dim = D>,
          Lin: TimeEvolution<Scalar = A, Dim = D> + TimeStep
{
    type Buffer = DiagRK4Buffer<NLin::Buffer, Lin::Buffer, Array<A, D>>;

    fn new_buffer(&self) -> Self::Buffer {
        DiagRK4Buffer {
            nlin: self.nlin.new_buffer(),
            lin: self.lin.new_buffer(),
            x: Array::zeros(self.lin.model_size()),
            lx: Array::zeros(self.lin.model_size()),
            k1: Array::zeros(self.lin.model_size()),
            k2: Array::zeros(self.lin.model_size()),
            k3: Array::zeros(self.lin.model_size()),
        }
    }
}

impl<A, D, NLin, Lin> TimeEvolution for DiagRK4<NLin, Lin>
    where A: Scalar,
          D: Dimension,
          NLin: SemiImplicitBuf<Scalar = A, Dim = D>,
          Lin: TimeEvolution<Scalar = A, Dim = D> + TimeStep<Time = A::Real>
{
    type Scalar = A;

    fn iterate<'a, S>(&self,
                      x: &'a mut ArrayBase<S, Self::Dim>,
                      mut buf: &mut Self::Buffer)
                      -> &'a mut ArrayBase<S, Self::Dim>
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
        buf.x.zip_mut_with(x, |buf, x| *buf = *x);
        buf.lx.zip_mut_with(x, |buf, lx| *buf = *lx);
        l.iterate(&mut buf.lx, &mut buf.lin);
        let mut k1 = f.nlin(x, &mut buf.nlin);
        buf.k1.zip_mut_with(k1, |buf, k1| *buf = *k1);
        Zip::from(&mut *k1)
            .and(&buf.x)
            .apply(|k1, &x_| { *k1 = x_ + k1.mul_real(dt_2); });
        let mut k2 = f.nlin(l.iterate(k1, &mut buf.lin), &mut buf.nlin);
        buf.k2.zip_mut_with(k2, |buf, k| *buf = *k);
        Zip::from(&mut *k2)
            .and(&buf.lx)
            .apply(|k2, &lx| { *k2 = lx + k2.mul_real(dt_2); });
        let mut k3 = f.nlin(k2, &mut buf.nlin);
        buf.k3.zip_mut_with(k3, |buf, k| *buf = *k);
        Zip::from(&mut *k3)
            .and(&buf.lx)
            .apply(|k3, &lx| { *k3 = lx + k3.mul_real(dt); });
        let mut k4 = f.nlin(l.iterate(k3, &mut buf.lin), &mut buf.nlin);
        Zip::from(&mut buf.x)
            .and(&buf.k1)
            .apply(|x_, k1_| *x_ = *x_ + k1_.mul_real(dt_6));
        l.iterate(&mut buf.x, &mut buf.lin);
        Zip::from(&mut buf.x)
            .and(&buf.k2)
            .and(&buf.k3)
            .apply(|x_, &k2_, &k3_| *x_ = *x_ + (k2_ + k3_).mul_real(dt_3));
        l.iterate(&mut buf.x, &mut buf.lin);
        Zip::from(&mut *k4)
            .and(&buf.x)
            .apply(|k4, &x_| { *k4 = x_ + k4.mul_real(dt_6); });
        k4
    }
}
