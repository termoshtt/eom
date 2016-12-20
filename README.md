ndarray-odeint [![Crate](http://meritbadge.herokuapp.com/ndarray-odeint)](https://crates.io/crates/ndarray-odeint) [![docs.rs](https://docs.rs/ndarray-odeint/badge.svg)](https://docs.rs/ndarray-odeint) [![Build Status](https://travis-ci.org/termoshtt/ndarray-odeint.svg?branch=master)](https://travis-ci.org/termoshtt/ndarray-odeint)
===============

solve ODE with rust-ndarray

Example
--------

```rust
use itertools::iterate;
use ndarray_odeint::lorenz63 as l63;

let dt = 0.01;
let p = l63::Parameter::default();
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

See issues

- [Explicit Schemes](/../../issues/2)
- [Implicit Schemes](/../../issues/3)

License
-------
MIT-License, see [LICENSE](LICENSE) file.
