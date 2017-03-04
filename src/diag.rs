
use super::traits::{TimeEvolution, OdeScalar};
use ndarray::*;
use std::marker::PhantomData;

/// Linear ODE with diagonalized matrix (exactly solvable)
pub struct Diagonal<A: OdeScalar<f64>, D: Dimension> {
    diag: Vec<A>,
    dt: f64,
    phantom: PhantomData<D>,
}

impl<A: OdeScalar<f64>, D: Dimension> Diagonal<A, D> {
    pub fn new(diag_of_matrix: RcArray<A, D>, dt: f64) -> Self {
        Diagonal {
            diag: diag_of_matrix.iter()
                .map(|x| (*x * dt).exp())
                .collect(),
            dt: dt,
            phantom: PhantomData,
        }
    }
}

impl<A: OdeScalar<f64>, D: Dimension> TimeEvolution<A, D> for Diagonal<A, D> {
    fn iterate(&self, mut x: RcArray<A, D>) -> RcArray<A, D> {
        for (val, d) in x.iter_mut().zip(self.diag.iter()) {
            *val *= *d;
        }
        x
    }
    fn get_dt(&self) -> f64 {
        self.dt
    }
}
