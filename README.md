ndarray-odeint [![Crate](http://meritbadge.herokuapp.com/ndarray-odeint)](https://crates.io/crates/ndarray-odeint) [![docs.rs](https://docs.rs/ndarray-odeint/badge.svg)](https://docs.rs/ndarray-odeint) [![Build Status](https://travis-ci.org/termoshtt/ndarray-odeint.svg?branch=master)](https://travis-ci.org/termoshtt/ndarray-odeint)
===============

solve ODE with rust-ndarray

Algorithms
-----------

- explicit
  - Euler
  - Heun
  - classical 4th order Runge-Kutta
- semi-implicit
  - stiff RK4

Example
--------

```rust
use ndarray::rcarr1;
use itertools::iterate;
use ndarray_odeint::prelude::*;

let dt = 0.01;
let eom = Lorenz63::default();
let teo = explicit::rk4(eom, dt);
let ts = iterate(rcarr1(&[1.0, 0.0, 0.0]), |y| teo.iterate(y.clone()));
let end_time = 10000;
for v in ts.take(end_time) {
  println!("{:?}", &v);
}
```

![Lorenz63 Attractor](lorenz63.png)

Figures are plotted by matplotlib (see [plot script](figure.py)).

Lyapunov Analysis
------------------
See [Notebook](CLV.ipynb)

Accuracy Check
---------------
See [Notebook](accuracy.ipynb)

License
-------
MIT-License, see [LICENSE](LICENSE) file.
