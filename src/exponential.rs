
use num_complex::Complex;
use num_traits::Float;

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
