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

impl<'a, A, F, D> TimeEvolution<A, OwnedRcRepr<A>, D> for &'a $($mut_),* DiagRK4<A, F, D>
    where A: OdeScalar<f64> + Exponential,
          for<'b> &'b $($mut_),* F: EOM<A, OwnedRcRepr<A>, D>,
          D: Dimension
{
    fn iterate(self, x: RcArray<A, D>) -> RcArray<A, D> {
        // constants
        let dt = self.dt;
        let dt_2 = 0.5 * self.dt;
        let dt_3 = self.dt / 3.0;
        let dt_6 = self.dt / 6.0;
        // operators
        let l = &self.lin_half;
        let f = &$($mut_),* self.f;
        // calc
        let k1 = f.rhs(x.clone());
        let l1 = l.iterate(k1.clone() * dt_2 + &x);
        let k2 = f.rhs(l1);
        let lx = l.iterate(x.clone());
        let l2 = k2.clone() * dt_2 + &lx;
        let k3 = f.rhs(l2);
        let l3 = l.iterate(lx + &k3 * dt);
        let k4 = f.rhs(l3);
        l.iterate(l.iterate(x + k1 * dt_6) + (k2 + k3) * dt_3) + k4 * dt_6
    }
}

}} // impl_time_evolution!

impl_time_evolution!();
impl_time_evolution!(mut);
