//! Fundamental traits for ODE

use ndarray::*;
use ndarray_linalg::*;

pub trait ModelSize<D: Dimension> {
    fn model_size(&self) -> D::Pattern;
}

/// Interface for time-step
pub trait TimeStep {
    type Time: RealScalar;
    fn get_dt(&self) -> Self::Time;
    fn set_dt(&mut self, dt: Self::Time);
}

/// Equation of motion (Explicit)
pub trait Explicit<S, D>: ModelSize<D>
    where S: DataMut,
          D: Dimension
{
    type Scalar: Scalar;
    type Time: RealScalar;
    /// calculate right hand side (rhs) of Explicit from current state
    fn rhs<'a>(&self, &'a mut ArrayBase<S, D>) -> &'a mut ArrayBase<S, D>;
}

pub trait SemiImplicit<S, D>: ModelSize<D>
    where S: DataMut,
          D: Dimension
{
    type Scalar: Scalar;
    type Time: RealScalar;
    /// non-linear part of stiff equation
    fn nlin<'a>(&self, &'a mut ArrayBase<S, D>) -> &'a mut ArrayBase<S, D>;
}

/// Time-evolution operator
pub trait TimeEvolutionBase<S, D>: ModelSize<D> + TimeStep
    where S: DataMut,
          D: Dimension
{
    type Scalar: Scalar;
    /// calculate next step
    fn iterate<'a>(&self, &'a mut ArrayBase<S, D>) -> &'a mut ArrayBase<S, D>;
}

pub trait TimeEvolution<A, D>
    : TimeEvolutionBase<OwnedRepr<A>, D, Scalar = A, Time = A::Real>
    + TimeEvolutionBase<OwnedRcRepr<A>, D, Scalar = A, Time = A::Real>
    + for<'a> TimeEvolutionBase<ViewRepr<&'a mut A>, D, Scalar = A, Time = A::Real>
    where A: Scalar,
          D: Dimension
{
}

/// Time-evolution operator with buffer
pub trait TimeEvolutionBufferedBase<S, D, Buffer>: ModelSize<D> + TimeStep
    where S: DataMut,
          D: Dimension
{
    type Scalar: Scalar;
    /// calculate next step
    fn iterate_buf<'a>(&self, &'a mut ArrayBase<S, D>, &mut Buffer) -> &'a mut ArrayBase<S, D>;
}

pub trait TimeEvolutionBuffered<A, D, Buffer>
    : TimeEvolutionBufferedBase<OwnedRepr<A>, D, Buffer, Scalar = A, Time = A::Real>
    + TimeEvolutionBufferedBase<OwnedRcRepr<A>, D, Buffer, Scalar = A, Time = A::Real>
    + for<'a> TimeEvolutionBufferedBase<ViewRepr<&'a mut A>, D, Buffer, Scalar = A, Time = A::Real>
    where A: Scalar,
          D: Dimension
{
}
