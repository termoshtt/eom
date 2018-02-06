extern crate eom;
extern crate ndarray;
extern crate ndarray_linalg;
extern crate num_traits;

use std::fs::*;
use std::io::Write;
use ndarray::*;
use ndarray_linalg::*;
use num_traits::int::PrimInt;

use eom::*;
use eom::traits::*;

fn accuracy<A, D, F, Sc>(mut teo: Sc, init: Array<A, D>) -> Vec<(Sc::Time, A::Real)>
where
    A: Scalar,
    D: Dimension,
    Sc: Scheme<F, Scalar = A, Dim = D>,
{
    let data: Vec<_> = (0..12)
        .map(|n| {
            let dt = into_scalar(0.01 / 2.pow(n) as f64);
            let t = 1000 * 2.pow(n);
            teo.set_dt(dt);
            let mut ts = adaptor::time_series(init.clone(), &mut teo);
            (dt, ts.nth(t - 1).unwrap())
        })
        .collect();
    data.windows(2)
        .map(|w| {
            let dt = w[0].0;
            let dev = (&w[1].1 - &w[0].1).norm();
            (dt, dev)
        })
        .collect()
}

fn check_accuracy<A, D, F, Sc>(teo: Sc, init: Array<A, D>, fname: &str)
where
    A: Scalar<Real = f64>,
    D: Dimension,
    Sc: Scheme<F, Scalar = A, Dim = D, Time = f64>,
{
    let acc = accuracy(teo, init);
    let mut f = File::create(fname).unwrap();
    write!(&mut f, "dt,dev\n").unwrap();
    for &(dt, dev) in acc.iter() {
        write!(&mut f, "{:.08e},{:.08e}\n", dt, dev).unwrap();
    }
}

fn main() {
    let l63 = ode::Lorenz63::default();
    check_accuracy(
        explicit::Euler::new(l63.clone(), 1.0),
        arr1(&[1.0, 0.0, 0.0]),
        "euler.csv",
    );
    check_accuracy(
        explicit::Heun::new(l63.clone(), 1.0),
        arr1(&[1.0, 0.0, 0.0]),
        "heun.csv",
    );
    check_accuracy(
        explicit::RK4::new(l63.clone(), 1.0),
        arr1(&[1.0, 0.0, 0.0]),
        "rk4.csv",
    );
    check_accuracy(
        semi_implicit::DiagRK4::new(l63.clone(), 1.0),
        arr1(&[1.0, 0.0, 0.0]),
        "diag_rk4.csv",
    );
}
