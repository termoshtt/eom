//! Fundamental traits for ODE

use ndarray::*;
use ndarray_linalg::*;

pub trait ModelSize<D: Dimension> {
    fn model_size(&self) -> D::Pattern;
}

/// Equation of motion (Explicit)
pub trait Explicit<S, D>: ModelSize<D>
    where S: DataMut,
          D: Dimension
{
    /// calculate right hand side (rhs) of Explicit from current state
    fn rhs<'a>(&self, &'a mut ArrayBase<S, D>) -> &'a mut ArrayBase<S, D>;
}

pub trait SemiImplicitDiag<Sn, Sd, D>: ModelSize<D>
    where Sn: DataMut,
          Sd: Data,
          D: Dimension
{
    /// non-linear part of stiff equation
    fn nlin<'a>(&self, &'a mut ArrayBase<Sn, D>) -> &'a mut ArrayBase<Sn, D>;
    /// Linear part of Explicit (assume to be diagonalized)
    fn diag(&self) -> ArrayBase<Sd, D>;
}

/// Time-evolution operator
pub trait TimeEvolutionBase<S, D>: ModelSize<D>
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

pub trait TimeEvolution<A, D>
    : TimeEvolutionBase<OwnedRepr<A>, D, Time = A::Real>
    + TimeEvolutionBase<OwnedRcRepr<A>, D, Time = A::Real>
    + for<'a> TimeEvolutionBase<ViewRepr<&'a mut A>, D, Time = A::Real>
    + TimeStep<Time = A::Real>
    where A: Scalar,
          D: Dimension
{
}

impl<A, D, EOM> TimeEvolution<A, D> for EOM
    where A: Scalar,
          D: Dimension,
          EOM: TimeEvolutionBase<OwnedRepr<A>, D, Time = A::Real>
                   + TimeEvolutionBase<OwnedRcRepr<A>, D, Time = A::Real>
                   + for<'a> TimeEvolutionBase<ViewRepr<&'a mut A>, D, Time = A::Real>
                   + TimeStep<Time = A::Real>
{
}
