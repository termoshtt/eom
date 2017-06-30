//! Fundamental traits for ODE

use ndarray::*;

/// Equation of motion (EOM)
pub trait EOM<A, S, D>
    where S: DataMut<Elem = A>,
          D: Dimension
{
    /// calculate right hand side (rhs) of EOM from current state
    fn rhs(self, &mut ArrayBase<S, D>) -> &mut ArrayBase<S, D>;
}

/// non-linear part of stiff equation
pub trait NonLinear<A, S, D>
    where S: DataMut<Elem = A>,
          D: Dimension
{
    fn nlin(self, &mut ArrayBase<S, D>) -> &mut ArrayBase<S, D>;
}

/// Diagonalized linear part of stiff equation
pub trait Diag<A, D>
    where D: Dimension
{
    /// Linear part of EOM (assume to be diagonalized)
    fn diagonal(&self) -> RcArray<A, D>;
}

/// Time-evolution operator
pub trait TimeEvolution<A, S, D>
    where S: DataMut<Elem = A>,
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
