
#[macro_use]
extern crate derive_new;
extern crate ndarray;
extern crate ndarray_linalg;
extern crate itertools;
extern crate num_complex;
extern crate num_traits;

pub mod traits;
pub mod prelude;
pub mod exponential;

// solvers
pub mod explicit;
pub mod diag;
pub mod semi_implicit;

// models
pub mod lorenz63;
pub mod lorenz96;
pub mod roessler;
pub mod goy_shell;
