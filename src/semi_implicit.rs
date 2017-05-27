//! Define semi-implicit schemes

use super::traits::*;
use super::diag::Diagonal;

use ndarray::*;

pub struct DiagRK4<A, F, D>
    where A: OdeScalar<f64> + Exponential,
          D: Dimension
{
    f: F,
    lin_half: Diagonal<A, D>,
    dt: f64,
}

pub fn diag_rk4<A, F, D>(f: F, dt: f64) -> DiagRK4<A, F, D>
    where A: OdeScalar<f64> + Exponential,
          F: Diag<A, D>,
          D: Dimension
{
    DiagRK4::new(f, dt)
}

impl<A, F, D> DiagRK4<A, F, D>
    where A: OdeScalar<f64> + Exponential,
          D: Dimension
{
    pub fn new(f: F, dt: f64) -> Self
        where F: Diag<A, D>
    {
        let diag = f.diagonal();
        let lin_half = Diagonal::new(diag, dt / 2.0);
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
    where A: OdeScalar<f64> + Exponential,
          S: DataMut<Elem=A> + DataClone<Elem=A>,
          for<'b> &'b $($mut_),* F: EOM<A, S, D>,
          D: Dimension
{
    fn iterate(self, x: ArrayBase<S, D>) -> ArrayBase<S, D> {
        // constants
        let dt = self.dt;
        let dt_2 = 0.5 * self.dt;
        let dt_3 = self.dt / 3.0;
        let dt_6 = self.dt / 6.0;
        // operators
        let l = &self.lin_half;
        let f = &$($mut_),* self.f;
        // calc
        let mut x_ = x.to_owned();
        let lx = l.iterate(x.clone());
        let mut k1 = f.rhs(x);
        let k1_ = k1.to_owned();
        azip!(mut k1, x_ in { *k1 = x_ + *k1 * dt_2 });
        let mut k2 = f.rhs(l.iterate(k1));
        let k2_ = k2.to_owned();
        azip!(mut k2, lx in { *k2 = lx + *k2 * dt_2 });
        let mut k3 = f.rhs(k2);
        let k3_ = k3.to_owned();
        azip!(mut k3, lx in { *k3 = lx + *k3 * dt });
        let mut k4 = f.rhs(l.iterate(k3));
        azip!(mut x_, k1_ in { *x_ = *x_ + k1_ * dt_6 });
        let mut x_ = l.iterate(x_);
        azip!(mut x_, k2_, k3_ in { *x_ = *x_ + (k2_ + k3_) * dt_3 });
        let x_ = l.iterate(x_);
        azip!(mut k4, x_ in { *k4 = x_ + *k4 * dt_6 });
        k4
    }
}

}} // impl_time_evolution!

impl_time_evolution!();
impl_time_evolution!(mut);
