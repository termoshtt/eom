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
pub trait Explicit<D: Dimension>: ModelSize<D> {
    type Scalar: Scalar;
    type Time: RealScalar;
    /// calculate right hand side (rhs) of Explicit from current state
    fn rhs<S>(&self, &mut ArrayBase<S, D>) where S: DataMut<Elem = Self::Scalar>;
}

pub trait SemiImplicit<D: Dimension>: ModelSize<D> {
    type Scalar: Scalar;
    type Time: RealScalar;
    /// non-linear part of stiff equation
    fn nlin<S>(&self, &mut ArrayBase<S, D>) where S: DataMut<Elem = Self::Scalar>;
}

/// Time-evolution operator with buffer
pub trait TimeEvolution<D: Dimension>: ModelSize<D> + TimeStep {
    type Scalar: Scalar;
    type Buffer;
    /// calculate next step
    fn iterate<S>(&self, &mut ArrayBase<S, D>, &mut Buffer) where S: DataMut<Elem = Self::Scalar>;
}
