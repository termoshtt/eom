//! Fundamental traits for solving EoM

use ndarray::*;
use ndarray_linalg::*;

/// Model specification
pub trait ModelSpec: Clone {
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

/// Core implementation for explicit schemes
pub trait Explicit: ModelSpec {
    /// calculate right hand side (rhs) of Explicit from current state
    fn rhs<'a, S>(&mut self, &'a mut ArrayBase<S, Self::Dim>) -> &'a mut ArrayBase<S, Self::Dim>
    where
        S: DataMut<Elem = Self::Scalar>;
}

/// Core implementation for semi-implicit schemes
pub trait SemiImplicit: ModelSpec {
    /// non-linear part of stiff equation
    fn nlin<'a, S>(&mut self, &'a mut ArrayBase<S, Self::Dim>) -> &'a mut ArrayBase<S, Self::Dim>
    where
        S: DataMut<Elem = Self::Scalar>;
    /// diagonal elements of stiff linear part
    fn diag(&self) -> Array<Self::Scalar, Self::Dim>;
}

/// Time-evolution operator
pub trait TimeEvolution: ModelSpec + TimeStep {
    /// calculate next step
    fn iterate<'a, S>(
        &mut self,
        &'a mut ArrayBase<S, Self::Dim>,
    ) -> &'a mut ArrayBase<S, Self::Dim>
    where
        S: DataMut<Elem = Self::Scalar>;

    /// calculate n-step
    fn iterate_n<'a, S>(
        &mut self,
        a: &'a mut ArrayBase<S, Self::Dim>,
        n: usize,
    ) -> &'a mut ArrayBase<S, Self::Dim>
    where
        S: DataMut<Elem = Self::Scalar>,
    {
        for _ in 0..n {
            self.iterate(a);
        }
        a
    }
}

/// Time evolution schemes
pub trait Scheme: TimeEvolution {
    type Core: ModelSpec<Scalar = Self::Scalar, Dim = Self::Dim>;
    /// Initialize with a core implementation
    fn new(f: Self::Core, dt: Self::Time) -> Self;
    /// Get immutable core
    fn core(&self) -> &Self::Core;
    /// Get mutable core
    fn core_mut(&mut self) -> &mut Self::Core;
}
