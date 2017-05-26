//! solve ODE with rust-ndarray
//!
//! Algorithms
//! -----------
//!
//! - explicit
//!   - Euler
//!   - Heun
//!   - classical 4th order Runge-Kutta
//! - semi-implicit
//!   - stiff RK4
//!
//! Models
//! -------
//! Basic chaotic dynamics are implemented as examples in [model submodule](model/)
//!
//! - [Lorenz three-variables system](https://en.wikipedia.org/wiki/Lorenz_system): [Lorenz63](model/lorenz63/struct.Lorenz63.html)
//! - [Lorenz 96 system](https://en.wikipedia.org/wiki/Lorenz_96_system): [Lorenz96](model/lorenz96/struct.Lorenz96.html)
//! - [Roessler system](https://en.wikipedia.org/wiki/Lorenz_syste://en.wikipedia.org/wiki/R%C3%B6ssler_attractor): [Roessler](model/roessler/struct.Roessler.html)
//! - GOY shell model: [GoyShell](model/goy_shell/struct.GoyShell.html)

#[macro_use]
extern crate derive_new;
extern crate ndarray;
extern crate ndarray_linalg;
extern crate itertools;
extern crate num_complex;
extern crate num_traits;

pub mod traits;
pub mod explicit;
pub mod diag;
pub mod semi_implicit;
pub mod model;
pub mod lyapunov;

pub use self::traits::*;
