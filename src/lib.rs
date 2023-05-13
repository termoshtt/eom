//! Configurable ODE/PDE solver

pub mod adaptor;
pub mod explicit;
pub mod lyapunov;
pub mod ode;
pub mod pde;
pub mod semi_implicit;

mod traits;
pub use traits::*;
