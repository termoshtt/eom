
use std::marker::PhantomData;
use ndarray::prelude::*;
use super::traits::{EOM, TimeEvolution};

pub struct Euler<F: EOM<D>, D: Dimension> {
    f: F,
    dt: f64,
    phantom: PhantomData<D>,
}

impl<F: EOM<D>, D: Dimension> Euler<F, D> {
    pub fn new(f: F, dt: f64) -> Self {
        Euler {
            f: f,
            dt: dt,
            phantom: PhantomData,
        }
    }
}

impl<F: EOM<D>, D: Dimension> TimeEvolution<D> for Euler<F, D> {
    #[inline(always)]
    fn iterate(&self, x: RcArray<f64, D>) -> RcArray<f64, D> {
        let fx = self.f.rhs(x.clone());
        x + fx * self.dt
    }
    #[inline(always)]
    fn get_dt(&self) -> f64 {
        self.dt
    }
}
