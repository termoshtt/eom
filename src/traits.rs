
use ndarray::{RcArray, Dimension, LinalgScalar};
use std::ops::*;

/// Equation of motion (EOM)
pub trait EOM<A, D>
    where D: Dimension
{
    /// calculate right hand side (rhs) of EOM from current state
    fn rhs(self, RcArray<A, D>) -> RcArray<A, D>;
}

/// Stiff equation with diagonalized linear part
pub trait Diag<A, D>
    where D: Dimension
{
    /// Linear part of EOM (assume to be diagonalized)
    fn diagonal(&self) -> RcArray<A, D>;
}

/// Time-evolution operator
pub trait TimeEvolution<A, D>
    where D: Dimension
{
    /// calculate next step
    fn iterate(self, RcArray<A, D>) -> RcArray<A, D>;
    fn get_dt(&self) -> f64;
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
