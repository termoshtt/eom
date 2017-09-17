Equation of Motions
====================
[![Crate](http://meritbadge.herokuapp.com/ndarray-odeint)](https://crates.io/crates/ndarray-odeint)
[![docs.rs](https://docs.rs/ndarray-odeint/badge.svg)](https://docs.rs/ndarray-odeint)
[![Build Status](https://travis-ci.org/termoshtt/ndarray-odeint.svg?branch=master)](https://travis-ci.org/termoshtt/ndarray-odeint)

![Lorenz63 Attractor](lorenz63.png)

`eom` consists of

- configurable ODE/PDE solver
  - Algorithms
    - explicit
      - Euler
      - Heun
      - classical 4th order Runge-Kutta
    - semi-implicit
      - stiff RK4
  - ODE
    - [Lorenz three-variables system](https://en.wikipedia.org/wiki/Lorenz_system)
    - [Lorenz 96 system](https://en.wikipedia.org/wiki/Lorenz_96_model)
    - [Roessler system](https://en.wikipedia.org/wiki/R%C3%B6ssler_attractor)
    - GOY shell model
  - PDE (under construction)
- Lyapunov analysis
  - [Lyapunov expoents of Lorenz 63 model](http://sprott.physics.wisc.edu/chaos/lorenzle.htm) / [code](examples/lyapunov.rs)
  - [Covarient Lyapunov vector (CLV)](https://arxiv.org/abs/1212.3961)/ [code](examples/clv.rs) / [notebook](CLV.ipynb)

License
-------
MIT-License, see [LICENSE](LICENSE) file.
