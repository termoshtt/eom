
use ndarray::{RcArray, Dimension};

/// Equation of motion (EOM)
pub trait EOM<D: Dimension> {
    /// calculate right hand side (rhs) of EOM from current state
    fn rhs(&self, RcArray<f64, D>) -> RcArray<f64, D>;
}

/// Stiff equation with diagonalized linear part
pub trait StiffDiag<D: Dimension>: EOM<D> {
    /// Non-Linear part of EOM
    fn nonlinear(&self, RcArray<f64, D>) -> RcArray<f64, D>;
    /// Linear part of EOM (assume to be diagonalized)
    fn linear_diagonal(&self) -> RcArray<f64, D>;

    fn rhs(&self, x: RcArray<f64, D>) -> RcArray<f64, D> {
        let nlin = self.nonlinear(x.clone());
        let a = self.linear_diagonal();
        nlin + a * x
    }
}

/// Time-evolution operator
pub trait TimeEvolution<D: Dimension> {
    /// calculate next step
    fn iterate(&self, RcArray<f64, D>) -> RcArray<f64, D>;
    fn get_dt(&self) -> f64;
}
