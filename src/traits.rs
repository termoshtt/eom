
use ndarray::{RcArray, Dimension};

/// calculate right hand side (rhs) of equation of motion from current state
pub trait EOM<D: Dimension> {
    fn rhs(&self, RcArray<f64, D>) -> RcArray<f64, D>;
}

/// Solve stiff equation.
/// The linear part is assumed to be diagonalized.
pub trait SemiImplicitDiag<D: Dimension> {
    /// Non-Linear part of EOM
    fn nonlinear(&self, RcArray<f64, D>) -> RcArray<f64, D>;
    /// Linear part of EOM (assume to be diagonalized)
    fn linear_diagonal(&self) -> Array<f64, D>;
}

/// calculate next step by integrating the equation of motion
pub trait TimeEvolution<D: Dimension> {
    fn iterate(&self, RcArray<f64, D>) -> RcArray<f64, D>;
    fn get_dt(&self) -> f64;
}
