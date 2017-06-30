//! Fundamental traits for ODE

use ndarray::*;

/// Equation of motion (EOM)
pub trait EOM<S, D>
    where S: DataMut,
          D: Dimension
{
    /// calculate right hand side (rhs) of EOM from current state
    fn rhs(self, &mut ArrayBase<S, D>) -> &mut ArrayBase<S, D>;
}

/// non-linear part of stiff equation
pub trait NonLinear<S, D>
    where S: DataMut,
          D: Dimension
{
    fn nlin(self, &mut ArrayBase<S, D>) -> &mut ArrayBase<S, D>;
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
    /// calculate next step
    fn iterate(self, &mut ArrayBase<S, D>) -> &mut ArrayBase<S, D>;
}

/// Interface for time-step
pub trait TimeStep<Time> {
    fn get_dt(&self) -> Time;
    fn set_dt(&mut self, dt: Time);
}
