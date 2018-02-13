//! solve ODE with rust-ndarray

#[macro_use]
extern crate derive_new;
extern crate fftw;
extern crate ndarray;
extern crate ndarray_linalg;
extern crate num_complex;
extern crate num_traits;

pub mod adaptor;
pub mod explicit;
pub mod lyapunov;
pub mod ode;
pub mod pde;
pub mod semi_implicit;
pub mod traits;
