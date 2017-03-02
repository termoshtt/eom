
extern crate ndarray;
extern crate ndarray_linalg;
extern crate itertools;

pub mod traits;
pub mod prelude;
pub mod lyapunov;

// solvers
pub mod explicit;
pub mod diag;
pub mod semi_implicit;

// models
pub mod lorenz63;
pub mod lorenz96;
pub mod roessler;
