//! Fundamental traits for ODE

use std::ops::*;
use num_complex::Complex;
use num_traits::Float;
use ndarray::*;

/// Equation of motion (EOM)
pub trait EOM<A, S, D>
    where S: Data<Elem = A>,
          D: Dimension
{
    /// calculate right hand side (rhs) of EOM from current state
    fn rhs(self, ArrayBase<S, D>) -> ArrayBase<S, D>;
}

/// non-linear part of stiff equation
pub trait NonLinear<A, S, D>
    where S: Data<Elem = A>,
          D: Dimension
{
    fn nlin(self, ArrayBase<S, D>) -> ArrayBase<S, D>;
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
    where S: Data<Elem = A>,
          D: Dimension
{
    /// calculate next step
    fn iterate(self, ArrayBase<S, D>) -> ArrayBase<S, D>;
}

/// Interface for time-step
pub trait TimeStep {
    fn get_dt(&self) -> f64;
    fn set_dt(&mut self, dt: f64);
}

/// utility trait for easy implementation
pub trait OdeScalar<R: Ring>: LinalgScalar + RMod<R> {}
impl<A, R: Ring> OdeScalar<R> for A where A: LinalgScalar + RMod<R> {}

/// Ring (math)
pub trait Ring
    : Add<Output = Self> + Sub<Output = Self> + Mul<Output = Self> + Sized {
}
impl<A> Ring for A where A: Add<Output = A> + Sub<Output = A> + Mul<Output = A> + Sized {}

/// R-module
pub trait RMod<R: Ring>: Mul<R, Output = Self> + Sized {}
impl<A, R: Ring> RMod<R> for A where A: Mul<R, Output = A> + Sized {}

/// exponential function
pub trait Exponential: Clone + Copy + Sized {
    fn exp(self) -> Self;
}

impl Exponential for f32 {
    fn exp(self) -> Self {
        <Self>::exp(self)
    }
}

impl Exponential for f64 {
    fn exp(self) -> Self {
        <Self>::exp(self)
    }
}

impl<T> Exponential for Complex<T>
    where T: Clone + Float
{
    fn exp(self) -> Self {
        <Complex<T>>::exp(&self)
    }
}
