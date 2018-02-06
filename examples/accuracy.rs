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

fn check_accuracy<A, D, F, Sc>(mut teo: Sc, init: Array<A, D>, fname: &str)
where
    A: Scalar<Real = f64>,
    D: Dimension,
    F: Explicit<Scalar = A, Dim = D>,
    Sc: Scheme<F, Scalar = A, Dim = D, Time = f64>,
{
    let data: Vec<_> = (0..12)
        .map(|n| {
            let dt = 0.01 / 2.pow(n) as f64;
            let t = 1000 * 2.pow(n);
            teo.set_dt(dt);
            let mut ts = adaptor::time_series(init.clone(), &mut teo);
            (dt, ts.nth(t - 1).unwrap())
        })
        .collect();

    let mut f = File::create(fname).unwrap();
    write!(&mut f, "dt,dev\n").unwrap();
    for i in 0..(data.len() - 1) {
        let dt = data[i].0;
        let dev = (&data[i + 1].1 - &data[i].1).norm();
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
}
