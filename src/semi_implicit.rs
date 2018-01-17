//! Define semi-implicit schemes

use ndarray::*;
use ndarray_linalg::*;

use super::traits::*;
use super::diag::{diagonal, Diagonal};

#[derive(Debug, Clone)]
pub struct DiagRK4<A, D, NLin, Lin>
where
    A: Scalar,
    D: Dimension,
    NLin: ModelSpec<Scalar = A, Dim = D>,
    Lin: ModelSpec<Scalar = A, Dim = D> + TimeStep<Time = A::Real>,
{
    nlin: NLin,
    lin: Lin,
    dt: Lin::Time,
    x: Array<A, D>,
    lx: Array<A, D>,
    k1: Array<A, D>,
    k2: Array<A, D>,
    k3: Array<A, D>,
}

pub fn diag_rk4<A, D, NLin>(nlin: NLin, dt: A::Real) -> DiagRK4<A, D, NLin, Diagonal<A, D>>
where
    A: Scalar,
    D: Dimension,
    NLin: StiffDiagonal<Scalar = A, Dim = D>,
{
    let lin = diagonal(&nlin, dt / into_scalar(2.0));
    let x = Array::zeros(lin.model_size());
    let lx = Array::zeros(lin.model_size());
    let k1 = Array::zeros(lin.model_size());
    let k2 = Array::zeros(lin.model_size());
    let k3 = Array::zeros(lin.model_size());
    DiagRK4 {
        nlin,
        lin,
        dt,
        x,
        lx,
        k1,
        k2,
        k3,
    }
}

impl<A, D, NLin, Lin> TimeStep for DiagRK4<A, D, NLin, Lin>
where
    A: Scalar,
    D: Dimension,
    NLin: ModelSpec<Scalar = A, Dim = D>,
    Lin: ModelSpec<Scalar = A, Dim = D> + TimeStep<Time = A::Real>,
{
    type Time = Lin::Time;

    fn get_dt(&self) -> Self::Time {
        self.dt
    }

    fn set_dt(&mut self, dt: Self::Time) {
        self.lin.set_dt(dt / into_scalar(2.0));
    }
}

impl<A, D, NLin, Lin> ModelSpec for DiagRK4<A, D, NLin, Lin>
where
    A: Scalar,
    D: Dimension,
    NLin: ModelSpec<Scalar = A, Dim = D>,
    Lin: ModelSpec<Scalar = A, Dim = D> + TimeStep<Time = A::Real>,
{
    type Scalar = A;
    type Dim = D;

    fn model_size(&self) -> <Self::Dim as Dimension>::Pattern {
        self.nlin.model_size() // TODO check
    }
}

impl<A, D, NLin, Lin> TimeEvolution for DiagRK4<A, D, NLin, Lin>
where
    A: Scalar,
    D: Dimension,
    NLin: SemiImplicit<Scalar = A, Dim = D>,
    Lin: TimeEvolution<Scalar = A, Dim = D> + TimeStep<Time = A::Real>,
{
    fn iterate<'a, S>(
        &mut self,
        x: &'a mut ArrayBase<S, Self::Dim>,
    ) -> &'a mut ArrayBase<S, Self::Dim>
    where
        S: DataMut<Elem = A>,
    {
        // constants
        let dt = self.dt;
        let dt_2 = self.dt / into_scalar(2.0);
        let dt_3 = self.dt / into_scalar(3.0);
        let dt_6 = self.dt / into_scalar(6.0);
        // operators
        let l = &mut self.lin;
        let f = &mut self.nlin;
        // calc
        self.x.zip_mut_with(x, |buf, x| *buf = *x);
        self.lx.zip_mut_with(x, |buf, lx| *buf = *lx);
        l.iterate(&mut self.lx);
        let k1 = f.nlin(x);
        self.k1.zip_mut_with(k1, |buf, k1| *buf = *k1);
        Zip::from(&mut *k1).and(&self.x).apply(|k1, &x_| {
            *k1 = x_ + k1.mul_real(dt_2);
        });
        let k2 = f.nlin(l.iterate(k1));
        self.k2.zip_mut_with(k2, |buf, k| *buf = *k);
        Zip::from(&mut *k2).and(&self.lx).apply(|k2, &lx| {
            *k2 = lx + k2.mul_real(dt_2);
        });
        let k3 = f.nlin(k2);
        self.k3.zip_mut_with(k3, |buf, k| *buf = *k);
        Zip::from(&mut *k3).and(&self.lx).apply(|k3, &lx| {
            *k3 = lx + k3.mul_real(dt);
        });
        let k4 = f.nlin(l.iterate(k3));
        Zip::from(&mut self.x)
            .and(&self.k1)
            .apply(|x_, k1_| *x_ = *x_ + k1_.mul_real(dt_6));
        l.iterate(&mut self.x);
        Zip::from(&mut self.x)
            .and(&self.k2)
            .and(&self.k3)
            .apply(|x_, &k2_, &k3_| *x_ = *x_ + (k2_ + k3_).mul_real(dt_3));
        l.iterate(&mut self.x);
        Zip::from(&mut *k4).and(&self.x).apply(|k4, &x_| {
            *k4 = x_ + k4.mul_real(dt_6);
        });
        k4
    }
}
