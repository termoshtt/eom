//! Fundamental traits

use ndarray::*;
use ndarray_linalg::*;

pub trait ModelSpec {
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
    type Scalar: Scalar;
    /// calculate right hand side (rhs) of Explicit from current state
    fn rhs<'a, S>(&self, &'a mut ArrayBase<S, Self::Dim>) -> &'a mut ArrayBase<S, Self::Dim>
        where S: DataMut<Elem = Self::Scalar>;
}

/// EoM for semi-implicit schemes
pub trait SemiImplicit: ModelSpec {
    type Scalar: Scalar;
    /// non-linear part of stiff equation
    fn nlin<'a, S>(&self, &'a mut ArrayBase<S, Self::Dim>) -> &'a mut ArrayBase<S, Self::Dim>
        where S: DataMut<Elem = Self::Scalar>;
}

/// EoM for semi-implicit schemes
pub trait SemiImplicitBuf: ModelSpec + BufferSpec {
    type Scalar: Scalar;
    /// non-linear part of stiff equation
    fn nlin<'a, S>(&self,
                   &'a mut ArrayBase<S, Self::Dim>,
                   &mut Self::Buffer)
                   -> &'a mut ArrayBase<S, Self::Dim>
        where S: DataMut<Elem = Self::Scalar>;
}

impl<F> SemiImplicitBuf for F
    where F: SemiImplicit + BufferSpec<Buffer = NoBuffer>
{
    type Scalar = F::Scalar;

    fn nlin<'a, S>(&self,
                   x: &'a mut ArrayBase<S, Self::Dim>,
                   _: &mut Self::Buffer)
                   -> &'a mut ArrayBase<S, Self::Dim>
        where S: DataMut<Elem = Self::Scalar>
    {
        self.nlin(x)
    }
}

/// EoM whose stiff linear part is diagonal
pub trait StiffDiagonal: SemiImplicit {
    /// diagonal elements of stiff linear part
    fn diag(&self) -> Array<Self::Scalar, Self::Dim>;
}

/// Time-evolution operator with buffer
pub trait TimeEvolution: BufferSpec + ModelSpec {
    type Scalar: Scalar;
    /// calculate next step
    fn iterate<'a, S>(&self,
                      &'a mut ArrayBase<S, Self::Dim>,
                      &mut Self::Buffer)
                      -> &'a mut ArrayBase<S, Self::Dim>
        where S: DataMut<Elem = Self::Scalar>;
}
