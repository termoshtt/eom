//! Fundamental traits for ODE

use ndarray::*;
use ndarray_linalg::*;

pub trait ModelSize {
    type Dim: Dimension;
    fn model_size(&self) -> <Self::Dim as Dimension>::Pattern;
}

pub trait WithBuffer {
    type Buffer;
    /// Generate new calculate buffer
    fn new_buffer(&self) -> Self::Buffer;
}

/// Interface for time-step
pub trait TimeStep {
    type Time: RealScalar;
    fn get_dt(&self) -> Self::Time;
    fn set_dt(&mut self, dt: Self::Time);
}

/// Equation of motion (Explicit)
pub trait Explicit: ModelSize {
    type Scalar: Scalar;
    /// calculate right hand side (rhs) of Explicit from current state
    fn rhs<'a, S>(&self, &'a mut ArrayBase<S, Self::Dim>) -> &'a mut ArrayBase<S, Self::Dim>
        where S: DataMut<Elem = Self::Scalar>;
}

pub trait SemiImplicit: ModelSize {
    type Scalar: Scalar;
    /// non-linear part of stiff equation
    fn nlin<'a, S>(&self, &'a mut ArrayBase<S, Self::Dim>) -> &'a mut ArrayBase<S, Self::Dim>
        where S: DataMut<Elem = Self::Scalar>;
}

/// Time-evolution operator with buffer
pub trait TimeEvolution: WithBuffer + ModelSize {
    type Scalar: Scalar;
    /// calculate next step
    fn iterate<'a, S>(&self,
                      &'a mut ArrayBase<S, Self::Dim>,
                      &mut Self::Buffer)
                      -> &'a mut ArrayBase<S, Self::Dim>
        where S: DataMut<Elem = Self::Scalar>;
}
