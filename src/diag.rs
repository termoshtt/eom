
use super::traits::TimeEvolution;
use ndarray::*;
use std::marker::PhantomData;

/// Linear ODE with diagonalized matrix (exactly solvable)
pub struct Diagonal<D: Dimension> {
    diag: Vec<f64>,
    dt: f64,
    phantom_dim: PhantomData<D>,
}

impl<D: Dimension> Diagonal<D> {
    pub fn new(diag_of_matrix: RcArray<f64, D>, dt: f64) -> Self {
        Diagonal {
            diag: diag_of_matrix.iter()
                .map(|x| (x * dt).exp())
                .collect(),
            dt: dt,
            phantom_dim: PhantomData,
        }
    }
}

impl<D: Dimension> TimeEvolution<D> for Diagonal<D> {
    fn iterate(&self, mut x: RcArray<f64, D>) -> RcArray<f64, D> {
        for (val, d) in x.iter_mut().zip(self.diag.iter()) {
            *val *= *d;
        }
        x
    }
    fn get_dt(&self) -> f64 {
        self.dt
    }
}
