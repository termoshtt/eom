
use ndarray::prelude::*;
use super::traits::{StiffDiag, TimeEvolution};
use super::diag::Diagonal;
use std::marker::PhantomData;

pub struct SemiImplicitDiagRK4<F: StiffDiag<D>, D: Dimension> {
    f: F,
    lin_half: Diagonal<D>,
    dt: f64,
    phantom_dim: PhantomData<D>,
}

impl<F: StiffDiag<D>, D: Dimension> SemiImplicitDiagRK4<F, D> {
    pub fn new(f: F, dt: f64) -> Self {
        let lin_half = Diagonal::new(f.linear_diagonal(), dt / 2.0);
        SemiImplicitDiagRK4 {
            f: f,
            lin_half: lin_half,
            dt: dt,
            phantom_dim: PhantomData,
        }
    }
}

impl<F: StiffDiag<D>, D: Dimension> TimeEvolution<D> for SemiImplicitDiagRK4<F, D> {
    fn iterate(&self, x: RcArray<f64, D>) -> RcArray<f64, D> {
        // constants
        let dt = self.dt;
        let dt_2 = 0.5 * self.dt;
        let v13 = 1.0 / 3.0;
        let v16 = 1.0 / 6.0;
        // operators
        let l = &self.lin_half;
        let f = &self.f;
        // calc
        let k1 = f.nonlinear(x.clone());
        let l1 = l.iterate(dt_2 * k1.clone() + &x);
        let k2 = f.nonlinear(l1);
        let lx = l.iterate(x.clone());
        let l2 = dt_2 * k2.clone() + &lx;
        let k3 = f.nonlinear(l2);
        let l3 = l.iterate(lx + dt * &k3);
        let k4 = f.nonlinear(l3);
        l.iterate(l.iterate(x + v16 * k1) + v13 * (k2 + k3)) + v16 * k4
    }
    fn get_dt(&self) -> f64 {
        self.dt
    }
}
