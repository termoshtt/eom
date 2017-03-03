
use ndarray::{RcArray, Dimension, LinalgScalar};
use std::ops::*;

/// Equation of motion (EOM)
pub trait EOM<A, D>
    where A: LinalgScalar,
          D: Dimension
{
    /// calculate right hand side (rhs) of EOM from current state
    fn rhs(&self, RcArray<A, D>) -> RcArray<A, D>;
}

/// Stiff equation with diagonalized linear part
pub trait StiffDiag<A, D>: EOM<A, D>
    where A: LinalgScalar,
          D: Dimension
{
    /// Non-Linear part of EOM
    fn nonlinear(&self, RcArray<A, D>) -> RcArray<A, D>;
    /// Linear part of EOM (assume to be diagonalized)
    fn linear_diagonal(&self) -> RcArray<A, D>;

    fn rhs(&self, x: RcArray<A, D>) -> RcArray<A, D> {
        let nlin = self.nonlinear(x.clone());
        let a = self.linear_diagonal();
        nlin + a * x
    }
}

/// Time-evolution operator
pub trait TimeEvolution<A, D>
    where A: LinalgScalar,
          D: Dimension
{
    /// calculate next step
    fn iterate(&self, RcArray<A, D>) -> RcArray<A, D>;
    fn get_dt(&self) -> f64;
}

pub trait Ring:
    Add<Output=Self> + Sub<Output=Self> + Mul<Output=Self>
    + AddAssign + SubAssign + MulAssign + Sized {}
impl <A> Ring for A
where A: Add<Output=A> + Sub<Output=A> + Mul<Output=A>
    + AddAssign + SubAssign + MulAssign + Sized {}

/// R-module
pub trait RMod<R: Ring>: Mul<R, Output = Self> + MulAssign<R> + Sized {}
impl<A, R: Ring> RMod<R> for A where A: Mul<R, Output = A> + MulAssign<R> + Sized {}

pub trait Exponential: Clone + Copy + Sized {
    fn exp_(&self) -> Self;
}

impl Exponential for f64 {
    fn exp_(&self) -> f64 {
        self.exp()
    }
}

pub trait OdeScalar<R: Ring>: LinalgScalar + Ring + RMod<R> + Exponential {}
impl<A, R: Ring> OdeScalar<R> for A where A: LinalgScalar + Ring + RMod<R> + Exponential {}
