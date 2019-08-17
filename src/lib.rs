/*!

Configurable ODE/PDE solver
===========================

This crate provides algorithms for solving ODE.

```
# use eom::traits::*;
# use eom::*;
# use ndarray::arr1;
let dt = 0.01;
// Setup Lorenz system (you may know as chaotic attractor system)
let eom = ode::Lorenz63::default();

// Setup classical Runge-Kutta method with Lorenz system.
let mut teo = explicit::RK4::new(eom, dt);

// Create Rust `Iterator`: the time series will be lazily evaluated in the iteration.
let ts = adaptor::time_series(arr1(&[1.0, 0.0, 0.0]), &mut teo);

// You can use any functions implemented for usual `Iterator`
for (t, v) in ts.take(1000).enumerate() {
    println!("{},{},{},{}", t as f64 * dt, v[0], v[1], v[2]);
}
```

How to implement ODE
---------------------
To define Lorenz-63 model, you need to create a struct
which implements the following traits:

- `ModelSpec` to determine the dimension of the system
- `Explicit` or `SemiImplicit` for time evolution

```
# use eom::traits::*;
# use ndarray::Ix1;
#[derive(Clone, Copy, Debug)]
pub struct Lorenz63 {
    pub p: f64,
    pub r: f64,
    pub b: f64,
}

impl ModelSpec for Lorenz63 {
    type Scalar = f64;   // We use double precision array
    type Dim = Ix1;      // We use one-dimensional array

    fn model_size(&self) -> usize {
        3   // Lorenz-63 is three dimensional
    }
}
```

This `ModelSpec` means we use `ndarray::Array<f64, Ix1>`
to contain the variables of Lorenz system, `X`, `Y`, and `Z`.

```
impl Explicit for Lorenz63 {
    fn rhs<'a, S>(&mut self, v: &'a mut ArrayBase<S, Ix1>) -> &'a mut ArrayBase<S, Ix1>
    where
        S: DataMut<Elem = f64>,
    {
        let x = v[0];
        let y = v[1];
        let z = v[2];
        v[0] = self.p * (y - x);
        v[1] = x * (self.r - z) - y;
        v[2] = x * y - self.b * z;
        v
    }
}
```

You need to implement the right-hand side (rhs) of the equation `dx/dt = f(x)`
where `x = [X, Y, Z]`.

*/

pub mod adaptor;
pub mod explicit;
pub mod lyapunov;
pub mod ode;
pub mod pde;
pub mod semi_implicit;
pub mod traits;
