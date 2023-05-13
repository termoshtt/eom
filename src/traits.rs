//! Fundamental traits for solving EoM

use ndarray::*;
use ndarray_linalg::*;
use num_traits::Float;

#[cfg(doc)]
use crate::{explicit::*, ode::*};

/// Model specification
///
/// To describe equations of motion,
/// we first have to specify the variable to describe the system state.
///
pub trait ModelSpec: Clone {
    type Scalar: Scalar;
    type Dim: Dimension;
    /// Number of scalars to describe the system state.
    fn model_size(&self) -> <Self::Dim as Dimension>::Pattern;
}

/// Interface for time-step
pub trait TimeStep {
    type Time: Scalar + Float;
    fn get_dt(&self) -> Self::Time;
    fn set_dt(&mut self, dt: Self::Time);
}

#[cfg_attr(doc, katexit::katexit)]
/// Explicit scheme for autonomous systems
///
/// Consider equation of motion of an autonomous system
/// described as an initial value problem of ODE:
/// $$
/// \frac{dx}{dt} = f(x),\space x(0) = x_0
/// $$
/// where $x = x(t)$ describes the system state specified by [ModelSpec] trait at a time $t$.
/// This trait specifies $f$ itself, i.e. this trait will be implemented for structs corresponds to
/// equations like [Lorenz63],
/// and used while implementing integrate algorithms like [Euler],
/// [Heun], and [RK4] algorithms.
/// Since these algorithms are independent from the detail of $f$,
/// this trait abstracts this part.
///
pub trait Explicit: ModelSpec {
    /// Evaluate $f(x)$ for a given state $x$
    ///
    /// This requires `&mut self` because evaluating $f$ may require
    /// additional internal memory allocated in `Self`.
    ///
    fn rhs<'a, S>(&mut self, x: &'a mut ArrayBase<S, Self::Dim>) -> &'a mut ArrayBase<S, Self::Dim>
    where
        S: DataMut<Elem = Self::Scalar>;
}

/// Core implementation for semi-implicit schemes
pub trait SemiImplicit: ModelSpec {
    /// non-linear part of stiff equation
    fn nlin<'a, S>(
        &mut self,
        x: &'a mut ArrayBase<S, Self::Dim>,
    ) -> &'a mut ArrayBase<S, Self::Dim>
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
        x: &'a mut ArrayBase<S, Self::Dim>,
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
