//! Example nonlinear ODEs

pub mod goy_shell;
pub mod lorenz63;
pub mod lorenz96;
pub mod roessler;

pub use self::goy_shell::GoyShell;
pub use self::lorenz63::Lorenz63;
pub use self::lorenz96::Lorenz96;
pub use self::roessler::Roessler;
