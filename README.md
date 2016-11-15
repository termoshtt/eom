ndarray-odeint [![Build Status](https://travis-ci.org/termoshtt/ndarray-odeint.svg?branch=master)](https://travis-ci.org/termoshtt/ndarray-odeint)
===============

solve ODE with rust-ndarray

Example
--------

```rust
extern crate ndarray;
extern crate ndarray_odeint;

use ndarray::prelude::*;

fn main() {
    let dt = 0.01;
    let l = |y| ndarray_odeint::lorenz63(10.0, 28.0, 8.0 / 3.0, y);
    let ts = ndarray_odeint::TimeSeries {
        teo: |y| ndarray_odeint::rk4(&l, dt, y),
        state: arr1(&[1.0, 0.0, 0.0]),
    };
    let end_time = 10000;
    println!("time,x,y,z");
    for (t, v) in ts.take(end_time).enumerate() {
        println!("{},{},{},{}", t as f64 * dt, v[0], v[1], v[2]);
    }
}
```

You can find complete code at [src/bin/main.rs](src/bin/main.rs)

![Lorenz63 Attractor](lorenz63.png)

This figure is plotted by matplotlib

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

  These needs linear algebra library for [rust-ndarray](https://github.com/bluss/rust-ndarray)

- semi-implicit scheme (for stiff equations)
  - to be done

[gnuplot]: http://www.gnuplot.info

License
-------
MIT-License, see [LICENSE](LICENSE) file.
