//! Fundamental traits

use ndarray::*;
use ndarray_linalg::*;

pub trait ModelSpec {
    type Scalar: Scalar;
    type Dim: Dimension;
    fn model_size(&self) -> <Self::Dim as Dimension>::Pattern;
}

/// Calculate with mutable buffer to keep `&self` interface
///
/// `&mut self` interface is too limited since it cannot be combined
/// with other `&self` functions even if re-generate the caluculation buffer.
pub trait BufferSpec {
    /// mutable state of caluculation
    type Buffer;
    /// Generate new calculate buffer
    fn new_buffer(&self) -> Self::Buffer;
}

/// Calculation can be done without buffer
pub type NoBuffer = ();

/// Interface for time-step
pub trait TimeStep {
    type Time: RealScalar;
    fn get_dt(&self) -> Self::Time;
    fn set_dt(&mut self, dt: Self::Time);
}

/// EoM for explicit schemes
pub trait Explicit: ModelSpec {
    /// calculate right hand side (rhs) of Explicit from current state
    fn rhs<'a, S>(&self, &'a mut ArrayBase<S, Self::Dim>) -> &'a mut ArrayBase<S, Self::Dim>
        where S: DataMut<Elem = Self::Scalar>;
}

/// EoM for explicit schemes
pub trait ExplicitBuf: ModelSpec + BufferSpec {
    /// calculate right hand side (rhs) of Explicit from current state
    fn rhs<'a, S>(&self,
                  &'a mut ArrayBase<S, Self::Dim>,
                  &mut Self::Buffer)
                  -> &'a mut ArrayBase<S, Self::Dim>
        where S: DataMut<Elem = Self::Scalar>;
}

/// EoM for semi-implicit schemes
pub trait SemiImplicit: ModelSpec {
    /// non-linear part of stiff equation
    fn nlin<'a, S>(&self, &'a mut ArrayBase<S, Self::Dim>) -> &'a mut ArrayBase<S, Self::Dim>
        where S: DataMut<Elem = Self::Scalar>;
}

/// EoM for semi-implicit schemes
pub trait SemiImplicitBuf: ModelSpec + BufferSpec {
    /// non-linear part of stiff equation
    fn nlin<'a, S>(&self,
                   &'a mut ArrayBase<S, Self::Dim>,
                   &mut Self::Buffer)
                   -> &'a mut ArrayBase<S, Self::Dim>
        where S: DataMut<Elem = Self::Scalar>;
}

/// EoM whose stiff linear part is diagonal
pub trait StiffDiagonal: ModelSpec {
    /// diagonal elements of stiff linear part
    fn diag(&self) -> Array<Self::Scalar, Self::Dim>;
}

/// Time-evolution operator with buffer
pub trait TimeEvolution: BufferSpec + ModelSpec {
    /// calculate next step
    fn iterate<'a, S>(&self,
                      &'a mut ArrayBase<S, Self::Dim>,
                      &mut Self::Buffer)
                      -> &'a mut ArrayBase<S, Self::Dim>
        where S: DataMut<Elem = Self::Scalar>;
}

impl<F> ExplicitBuf for F
    where F: Explicit + BufferSpec<Buffer = NoBuffer>
{
    fn rhs<'a, S>(&self,
                  x: &'a mut ArrayBase<S, Self::Dim>,
                  _: &mut Self::Buffer)
                  -> &'a mut ArrayBase<S, Self::Dim>
        where S: DataMut<Elem = Self::Scalar>
    {
        self.rhs(x)
    }
}

impl<F> SemiImplicitBuf for F
    where F: SemiImplicit + BufferSpec<Buffer = NoBuffer>
{
    fn nlin<'a, S>(&self,
                   x: &'a mut ArrayBase<S, Self::Dim>,
                   _: &mut Self::Buffer)
                   -> &'a mut ArrayBase<S, Self::Dim>
        where S: DataMut<Elem = Self::Scalar>
    {
        self.nlin(x)
    }
}
