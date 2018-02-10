//! Fundamental traits

use ndarray::*;
use ndarray_linalg::*;

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

/// EoM for explicit schemes
pub trait Explicit: ModelSpec {
    /// calculate right hand side (rhs) of Explicit from current state
    fn rhs<'a, S>(&mut self, &'a mut ArrayBase<S, Self::Dim>) -> &'a mut ArrayBase<S, Self::Dim>
    where
        S: DataMut<Elem = Self::Scalar>;
}

/// EoM for semi-implicit schemes
pub trait SemiImplicit: ModelSpec {
    /// non-linear part of stiff equation
    fn nlin<'a, S>(&mut self, &'a mut ArrayBase<S, Self::Dim>) -> &'a mut ArrayBase<S, Self::Dim>
    where
        S: DataMut<Elem = Self::Scalar>;
    /// diagonal elements of stiff linear part
    fn diag(&self) -> Array<Self::Scalar, Self::Dim>;
}

/// Time-evolution operator with buffer
pub trait TimeEvolution: ModelSpec + TimeStep {
    /// calculate next step
    fn iterate<'a, S>(
        &mut self,
        &'a mut ArrayBase<S, Self::Dim>,
    ) -> &'a mut ArrayBase<S, Self::Dim>
    where
        S: DataMut<Elem = Self::Scalar>;

    /// calculate next step
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
    fn new(f: Self::Core, dt: Self::Time) -> Self;
    fn core(&self) -> &Self::Core;
    fn core_mut(&mut self) -> &mut Self::Core;
}
