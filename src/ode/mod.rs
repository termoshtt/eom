//! Example nonlinear ODEs

mod goy_shell;
mod lorenz63;
mod lorenz96;
mod roessler;

pub use self::goy_shell::GoyShell;
pub use self::lorenz63::Lorenz63;
pub use self::lorenz96::Lorenz96;
pub use self::roessler::Roessler;
