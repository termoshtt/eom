Equation of Motions
====================
[![Crate](http://meritbadge.herokuapp.com/eom)](https://crates.io/crates/eom)
[![docs.rs](https://docs.rs/eom/badge.svg)](https://docs.rs/eom)
[![Build Status](https://travis-ci.org/termoshtt/eom.svg?branch=master)](https://travis-ci.org/termoshtt/eom)

![Lorenz63 Attractor](lorenz63.png)

configurable ODE/PDE solver
----------------------------
- Algorithms
  - explicit schemes
    - Euler
    - Heun
    - classical 4th order Runge-Kutta
  - semi-implicit schemes
    - stiff RK4
- ODE
  - [Lorenz three-variables system](https://en.wikipedia.org/wiki/Lorenz_system)
  - [Lorenz 96 system](https://en.wikipedia.org/wiki/Lorenz_96_model)
  - [Roessler system](https://en.wikipedia.org/wiki/R%C3%B6ssler_attractor)
  - GOY shell model
- PDE (under construction)

Lyapunov analysis
-----------------
- [Lyapunov expoents of Lorenz 63 model](http://sprott.physics.wisc.edu/chaos/lorenzle.htm)
  - [example](examples/lyapunov.rs)
- [Covarient Lyapunov vector (CLV)](https://arxiv.org/abs/1212.3961)
  - [example](examples/clv.rs) 
  - [notebook](CLV.ipynb)

License
-------
MIT-License, see [LICENSE](LICENSE) file.
