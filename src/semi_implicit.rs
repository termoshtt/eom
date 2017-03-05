
use super::traits::{StiffDiag, TimeEvolution, OdeScalar};
use super::diag::Diagonal;

use ndarray::{RcArray, Dimension};
use num_extra::Exponential;
use std::marker::PhantomData;

pub struct DiagRK4<A, F, D>
    where A: OdeScalar<f64> + Exponential,
          F: StiffDiag<A, D>,
          D: Dimension
{
    f: F,
    lin_half: Diagonal<A, D>,
    dt: f64,
    phantom_dim: PhantomData<D>,
}

pub fn diag_rk4<A, F, D>(f: F, dt: f64) -> DiagRK4<A, F, D>
    where A: OdeScalar<f64> + Exponential,
          F: StiffDiag<A, D>,
          D: Dimension
{
    DiagRK4::new(f, dt)
}

impl<A, F, D> DiagRK4<A, F, D>
    where A: OdeScalar<f64> + Exponential,
          F: StiffDiag<A, D>,
          D: Dimension
{
    pub fn new(f: F, dt: f64) -> Self {
        let lin_half = Diagonal::new(f.linear_diagonal(), dt / 2.0);
        DiagRK4 {
            f: f,
            lin_half: lin_half,
            dt: dt,
            phantom_dim: PhantomData,
        }
    }
}

impl<A, F, D> TimeEvolution<A, D> for DiagRK4<A, F, D>
    where A: OdeScalar<f64> + Exponential,
          F: StiffDiag<A, D>,
          D: Dimension
{
    fn iterate(&self, x: RcArray<A, D>) -> RcArray<A, D> {
        // constants
        let dt = self.dt;
        let dt_2 = 0.5 * self.dt;
        let dt_3 = self.dt / 3.0;
        let dt_6 = self.dt / 6.0;
        // operators
        let l = &self.lin_half;
        let f = &self.f;
        // calc
        let k1 = f.nonlinear(x.clone());
        let l1 = l.iterate(k1.clone() * dt_2 + &x);
        let k2 = f.nonlinear(l1);
        let lx = l.iterate(x.clone());
        let l2 = k2.clone() * dt_2 + &lx;
        let k3 = f.nonlinear(l2);
        let l3 = l.iterate(lx + &k3 * dt);
        let k4 = f.nonlinear(l3);
        l.iterate(l.iterate(x + k1 * dt_6) + (k2 + k3) * dt_3) + k4 * dt_6
    }
    fn get_dt(&self) -> f64 {
        self.dt
    }
}
