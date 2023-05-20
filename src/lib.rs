//! Configurable ODE solver
//!
//! Design
//! -------
//! When we try to solve initial value problem (IVP) of an ordinal differential equation (ODE),
//! we have to specify
//!
//! - the model space. Writing the ODE as a form $dx/dt = f(x)$,
//!   the linear space where $x$ belong to, e.g. $\mathbb{R}^n$ or $\mathbb{C}^n$
//!   is called model space, and represented by [ModelSpec] trait.
//! - the equation, i.e. $f$ of $dx/dt = f(x)$,
//!   e.g. Lorenz three variable equation, single or multiple pendulum and so on.
//! - the scheme, i.e. how to solve given ODE,
//!   e.g. explicit Euler, Runge-Kutta, symplectic Euler, and so on.
//!
//! Some equations requires some schemes.
//! For example, Hamilton systems require symplectic schemes,
//! or stiff equations require semi- or full-implicit schemes.
//! We would like to implement these schemes without fixing ODE,
//! but its abstraction depends on each scheme.
//! Explicit schemes assumes the ODE is in a form
//! $$
//! \frac{dx}{dt} = f(x, t)
//! $$
//! and hope to abstract $f$, but symplectic schemes assumes the ODE must be defined with Hamiltonian $H$
//! $$
//! \frac{\partial p}{\partial t} = -\frac{\partial H}{\partial q},
//! \frac{\partial q}{\partial t} = \frac{\partial H}{\partial p}.
//! $$
//! Some equation may be compatible several abstractions.
//! Hamiltonian systems can be integrated with explicit schemes
//! by ignoring phase-space volume contraction,
//! or stiff systems can be integrated with explicit schemes with very small time steps.
//!
//! This crate introduces traits for each abstractions, e.g. [Explicit] or [SemiImplicit],
//! which are implemented for each equations corresponds to ODE itself, e.g. [ode::Lorenz63].
//! Schemes, e.g. [explicit::Euler], use this traits as type-bound.
//!
//! <!--
//! KaTeX auto render
//! Crate-level proc-macro is currently unstable feature.
//! -->
//! <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.13.13/dist/katex.min.css" integrity="sha384-RZU/ijkSsFbcmivfdRBQDtwuwVqK7GMOw6IMvKyeWL2K5UAlyp6WonmB8m7Jd0Hn" crossorigin="anonymous">
//! <script defer src="https://cdn.jsdelivr.net/npm/katex@0.13.13/dist/katex.min.js" integrity="sha384-pK1WpvzWVBQiP0/GjnvRxV4mOb0oxFuyRxJlk6vVw146n3egcN5C925NCP7a7BY8" crossorigin="anonymous"></script>
//! <script defer src="https://cdn.jsdelivr.net/npm/katex@0.13.13/dist/contrib/auto-render.min.js" integrity="sha384-vZTG03m+2yp6N6BNi5iM4rW4oIwk5DfcNdFfxkk9ZWpDriOkXX8voJBFrAO7MpVl" crossorigin="anonymous"></script>
//! <script>
//!     document.addEventListener("DOMContentLoaded", function() {
//!         renderMathInElement(document.body, {
//!           // customised options
//!           // • auto-render specific keys, e.g.:
//!           delimiters: [
//!               {left: '$$', right: '$$', display: true},
//!               {left: '$', right: '$', display: false},
//!               {left: '\\(', right: '\\)', display: false},
//!               {left: '\\[', right: '\\]', display: true}
//!           ],
//!           // • rendering keys, e.g.:
//!           throwOnError : false
//!         });
//!     });
//! </script>
//!

pub mod adaptor;
pub mod explicit;
pub mod lyapunov;
pub mod ode;
pub mod pde;
pub mod semi_implicit;

mod traits;
pub use traits::*;
