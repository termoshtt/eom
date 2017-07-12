//! Fundamental traits for ODE

use ndarray::*;
use ndarray_linalg::*;

/// Equation of motion (EOM)
pub trait EOM<S, D>
    where S: DataMut,
          D: Dimension
{
    type Time: RealScalar;
    /// calculate right hand side (rhs) of EOM from current state
    fn rhs<'a>(&self, &'a mut ArrayBase<S, D>) -> &'a mut ArrayBase<S, D>;
}

/// non-linear part of stiff equation
pub trait NonLinear<S, D>
    where S: DataMut,
          D: Dimension
{
    type Time: RealScalar;
    fn nlin<'a>(&self, &'a mut ArrayBase<S, D>) -> &'a mut ArrayBase<S, D>;
}

/// Diagonalized linear part of stiff equation
pub trait Diag<S, D>
    where S: Data,
          D: Dimension
{
    /// Linear part of EOM (assume to be diagonalized)
    fn diagonal(&self) -> ArrayBase<S, D>;
}

/// Time-evolution operator
pub trait TimeEvolution<S, D>
    where S: DataMut,
          D: Dimension
{
    type Time: RealScalar;
    /// calculate next step
    fn iterate<'a>(&self, &'a mut ArrayBase<S, D>) -> &'a mut ArrayBase<S, D>;
}

/// Interface for time-step
pub trait TimeStep {
    type Time: RealScalar;
    fn get_dt(&self) -> Self::Time;
    fn set_dt(&mut self, dt: Self::Time);
}
