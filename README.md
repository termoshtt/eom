ndarray-odeint [![Build Status](https://travis-ci.org/termoshtt/ndarray-odeint.svg?branch=master)](https://travis-ci.org/termoshtt/ndarray-odeint)
===============

solve ODE with rust-ndarray

Example
--------

```rust
use itertools::iterate;
use ndarray_odeint::lorenz63 as l63;

let dt = 0.01;
let p = l63::default_parameter();
let l = |y| l63::f(p, y);
let ts = iterate(arr1(&[1.0, 0.0, 0.0]),
                 |y| ndarray_odeint::explicit::rk4(&l, dt, y.clone()));
let end_time = 10000;
for v in ts.take(end_time) {
  println!("{:?}", &v);
}
```

You can find complete code at [src/bin/main.rs](src/bin/main.rs)

![Lorenz63 Attractor](lorenz63.png)

This figure is plotted by matplotlib (see [plot script](figure.py)).

Developement status
--------------------

- explicit scheme
  - [x] forward Euler (euler)
  - [ ] refined Euler (reuler)
  - [ ] Heun (heun)
  - [x] classical Runge-Kutta (rk4)

- implicit scheme
  - [ ] backward Euler (beuler)
  - [ ] [diagonally implicit runge-kutta (dirk)](http://epubs.siam.org/doi/abs/10.1137/0714068)

- semi-implicit scheme (for stiff equations)
  - to be done

License
-------
MIT-License, see [LICENSE](LICENSE) file.
