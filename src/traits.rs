//! Fundamental traits for solving EoM

use ndarray::*;
use ndarray_linalg::*;
use num_traits::Float;

#[cfg(doc)]
use crate::{explicit::*, ode::*, semi_implicit::*};

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
/// Abstraction for implementing explicit schemes
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

#[cfg_attr(doc, katexit::katexit)]
/// Abstraction for implementing semi-implicit schemes for stiff equations
///
/// Consider equation of motion of a stiff autonomous system
/// described as an initial value problem of ODE:
/// $$
/// \frac{dx}{dt} = Ax + f(x),\space x(0) = x_0
/// $$
/// where $x = x(t)$ describes the system state specified by [ModelSpec] trait.
/// We split the right hand side of the equation
/// as the linear part $Ax$ to be stiff and the nonlinear part $f(x)$ not to be stiff.
/// In addition, we assume $A$ is diagonalizable,
/// and $x$ is selected to make $A$ diagonal.
/// Similar to [Explicit], this trait abstracts the pair $(A, f)$ to implement
/// semi-implicit schemes like [DiagRK4].
///
/// Stiff equations and semi-implicit schemes
/// -------------------------------------------
/// The stiffness causes numerical instabilities.
/// For example, consider solving one-dimensional ODE $dx/dt = -\lambda x$
/// with large $\lambda$ using explicit Euler scheme.
/// Apparently, the solution is $x(t) = x(0)e^{-\lambda t}$,
/// which converges to $0$ very quickly.
/// However, to capture this process using explicit scheme like Euler scheme,
/// we need as small $\Delta t$ as $\lambda^{-1}$.
/// Such small $\Delta t$ is usually unacceptable,
/// and implicit schemes are used for stiff equations,
/// but full implicit schemes require solving fixed point problem like
/// $1 + \lambda f(x) = 0$, which makes another instabilities.
/// Semi-implicit schemes has been introduced to resolve this situation,
/// i.e. use implicit scheme only for stiff linear part $Ax$
/// and use explicit schemes on non-stiff part $f(x)$.
///
pub trait SemiImplicit: ModelSpec {
    /// Non-stiff part $f(x)$
    fn nlin<'a, S>(
        &mut self,
        x: &'a mut ArrayBase<S, Self::Dim>,
    ) -> &'a mut ArrayBase<S, Self::Dim>
    where
        S: DataMut<Elem = Self::Scalar>;
    /// Diagonal elements of stiff linear part of $A$
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
