
use num_extra::{Ring, RMod};
use ndarray::{RcArray, Dimension, LinalgScalar};

/// Equation of motion (EOM)
pub trait EOM<A, D>
    where D: Dimension
{
    /// calculate right hand side (rhs) of EOM from current state
    fn rhs(&self, RcArray<A, D>) -> RcArray<A, D>;
}

/// Stiff equation with diagonalized linear part
pub trait StiffDiag<A, D>
    where D: Dimension
{
    /// Non-Linear part of EOM
    fn nonlinear(&self, RcArray<A, D>) -> RcArray<A, D>;
    /// Linear part of EOM (assume to be diagonalized)
    fn linear_diagonal(&self) -> RcArray<A, D>;
}

impl<A, D, F> EOM<A, D> for F
    where F: StiffDiag<A, D>,
          A: LinalgScalar,
          D: Dimension
{
    fn rhs(&self, x: RcArray<A, D>) -> RcArray<A, D> {
        let nlin = self.nonlinear(x.clone());
        let a = self.linear_diagonal();
        nlin + a * x
    }
}

/// Time-evolution operator
pub trait TimeEvolution<A, D>
    where D: Dimension
{
    /// calculate next step
    fn iterate(&self, RcArray<A, D>) -> RcArray<A, D>;
    fn get_dt(&self) -> f64;
}

/// utility trait for easy implementation
pub trait OdeScalar<R: Ring>: LinalgScalar + Ring + RMod<R> {}
impl<A, R: Ring> OdeScalar<R> for A where A: LinalgScalar + Ring + RMod<R> {}
