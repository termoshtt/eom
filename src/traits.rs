
use ndarray::RcArray;

/// calculate right hand side (rhs) of equation of motion from current state
pub trait RHS {
    type Dim;
    fn rhs(&self, RcArray<f64, Self::Dim>) -> RcArray<f64, Self::Dim>;
}

/// calculate next step by integrating the equation of motion
pub trait TimeEvolution {
    type Dim;
    fn iterate(&self, RcArray<f64, Self::Dim>) -> RcArray<f64, Self::Dim>;
    fn get_dt(&self) -> f64;
}
