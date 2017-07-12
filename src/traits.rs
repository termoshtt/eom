//! Fundamental traits for ODE

use ndarray::*;
use ndarray_linalg::*;

/// Equation of motion (Explicit)
pub trait Explicit<S, D>
    where S: DataMut,
          D: Dimension
{
    type Time: RealScalar;
    /// calculate right hand side (rhs) of Explicit from current state
    fn rhs<'a>(&self, &'a mut ArrayBase<S, D>) -> &'a mut ArrayBase<S, D>;
}

pub trait SemiImplicitDiag<Sn, Sd, D>
    where Sn: DataMut,
          Sd: Data,
          D: Dimension
{
    type Time: RealScalar;
    /// non-linear part of stiff equation
    fn nlin<'a>(&self, &'a mut ArrayBase<Sn, D>) -> &'a mut ArrayBase<Sn, D>;
    /// Linear part of Explicit (assume to be diagonalized)
    fn diag(&self) -> ArrayBase<Sd, D>;
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
