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

fn check_accuracy<Sc>(fname: &str)
where
    Sc: Scheme<ode::Lorenz63, Scalar = f64, Dim = Ix1, Time = f64>,
{
    let data: Vec<_> = (0..12)
        .map(|n| {
            let dt = 0.1 / 2.pow(n) as f64;
            let eom = ode::Lorenz63::default();
            let mut teo = Sc::new(eom, dt);
            let t = 100 * 2.pow(n);
            let ts = adaptor::time_series(arr1(&[1.0, 0.0, 0.0]), &mut teo);
            (dt, ts.take(t).last().unwrap())
        })
        .collect();

    let mut f = File::create(fname).unwrap();
    write!(&mut f, "dt,dev\n").unwrap();
    for i in 0..(data.len() - 1) {
        let dt = data[i].0;
        let dev = (&data[i + 1].1 - &data[i].1).norm();
        write!(&mut f, "{},{}\n", dt, dev).unwrap();
    }
}

fn main() {
    check_accuracy::<explicit::Euler<ode::Lorenz63>>("euler.csv");
    check_accuracy::<explicit::Heun<ode::Lorenz63>>("heun.csv");
    check_accuracy::<explicit::RK4<ode::Lorenz63>>("rk4.csv");
}
