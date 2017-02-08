
use ndarray::{RcArray, Dimension};

/// calculate right hand side (rhs) of equation of motion from current state
pub trait EOM<D: Dimension> {
    fn rhs(&self, RcArray<f64, D>) -> RcArray<f64, D>;
}

/// calculate next step by integrating the equation of motion
pub trait TimeEvolution<D: Dimension> {
    fn iterate(&self, RcArray<f64, D>) -> RcArray<f64, D>;
    fn get_dt(&self) -> f64;
}
