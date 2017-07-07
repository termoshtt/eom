ndarray-odeint
===============
[![Crate](http://meritbadge.herokuapp.com/ndarray-odeint)](https://crates.io/crates/ndarray-odeint)
[![docs.rs](https://docs.rs/ndarray-odeint/badge.svg)](https://docs.rs/ndarray-odeint)
[![Build Status](https://travis-ci.org/termoshtt/ndarray-odeint.svg?branch=master)](https://travis-ci.org/termoshtt/ndarray-odeint)

solve ODE with rust-ndarray

Algorithms
-----------

- explicit
  - Euler
  - Heun
  - classical 4th order Runge-Kutta
- semi-implicit
  - stiff RK4

Models
-------
 Basic chaotic dynamics are implemented as examples

- [Lorenz three-variables system](https://en.wikipedia.org/wiki/Lorenz_system)
- [Lorenz 96 system](https://en.wikipedia.org/wiki/Lorenz_96_model)
- [Roessler system](https://en.wikipedia.org/wiki/R%C3%B6ssler_attractor)
- GOY shell model

![Lorenz63 Attractor](lorenz63.png)

Figures are plotted by matplotlib (see [plot script](figure.py)).

Lyapunov Analysis
------------------

- [Lyapunov expoents of Lorenz 63 model](http://sprott.physics.wisc.edu/chaos/lorenzle.htm): [code](examples/lyapunov.rs)
- [CLV: covarient Lyapunov vector](https://arxiv.org/abs/1212.3961): [code](examples/clv.rs) [notebook](CLV.ipynb)

Accuracy Check
---------------
See [Notebook](accuracy.ipynb)

License
-------
MIT-License, see [LICENSE](LICENSE) file.
