
use ndarray::prelude::*;
use super::traits::{StiffDiag, TimeEvolution};
use std::marker::PhantomData;

pub struct SemiImplicitDiag<F: StiffDiag<D>, D: Dimension> {
    f: F,
    dt: f64,
    phantom_dim: PhantomData<D>,
}

impl<F: StiffDiag<D>, D: Dimension> TimeEvolution<D> for SemiImplicitDiag<F, D> {
    fn iterate(&self, x: RcArray<f64, D>) -> RcArray<f64, D> {
        x
    }
    fn get_dt(&self) -> f64 {
        self.dt
    }
}
