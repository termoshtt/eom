//! Fundamental traits

use ndarray::*;
use ndarray_linalg::*;

pub trait ModelSpec {
    type Scalar: Scalar;
    type Dim: Dimension;
    fn model_size(&self) -> <Self::Dim as Dimension>::Pattern;
}

/// Interface for time-step
pub trait TimeStep {
    type Time: RealScalar;
    fn get_dt(&self) -> Self::Time;
    fn set_dt(&mut self, dt: Self::Time);
}

/// EoM for explicit schemes
pub trait Explicit: ModelSpec + Clone {
    /// calculate right hand side (rhs) of Explicit from current state
    fn rhs<'a, S>(&mut self, &'a mut ArrayBase<S, Self::Dim>) -> &'a mut ArrayBase<S, Self::Dim>
    where
        S: DataMut<Elem = Self::Scalar>;
}

/// EoM for semi-implicit schemes
pub trait SemiImplicit: ModelSpec + Clone {
    /// non-linear part of stiff equation
    fn nlin<'a, S>(&mut self, &'a mut ArrayBase<S, Self::Dim>) -> &'a mut ArrayBase<S, Self::Dim>
    where
        S: DataMut<Elem = Self::Scalar>;
}

/// EoM whose stiff linear part is diagonal
pub trait StiffDiagonal: ModelSpec {
    /// diagonal elements of stiff linear part
    fn diag(&self) -> Array<Self::Scalar, Self::Dim>;
}

/// Time-evolution operator with buffer
pub trait TimeEvolution: ModelSpec + Clone {
    /// calculate next step
    fn iterate<'a, S>(
        &mut self,
        &'a mut ArrayBase<S, Self::Dim>,
    ) -> &'a mut ArrayBase<S, Self::Dim>
    where
        S: DataMut<Elem = Self::Scalar>;
}
