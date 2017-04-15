
use ndarray::*;
use std::marker::PhantomData;
use super::traits::{TimeEvolution, OdeScalar};
use super::exponential::Exponential;

/// Linear ODE with diagonalized matrix (exactly solvable)
pub struct Diagonal<A, D>
    where A: OdeScalar<f64> + Exponential,
          D: Dimension
{
    diag: Vec<A>,
    dt: f64,
    phantom: PhantomData<D>,
}

impl<A, D> Diagonal<A, D>
    where A: OdeScalar<f64> + Exponential,
          D: Dimension
{
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

impl<'a, A, D> TimeEvolution<A, D> for &'a Diagonal<A, D>
    where A: OdeScalar<f64> + Exponential,
          D: Dimension
{
    fn iterate(self, mut x: RcArray<A, D>) -> RcArray<A, D> {
        for (val, d) in x.iter_mut().zip(self.diag.iter()) {
            *val = *val * *d;
        }
        x
    }
    fn get_dt(&self) -> f64 {
        self.dt
    }
}
