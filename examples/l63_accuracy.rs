extern crate eom;
extern crate ndarray;
extern crate ndarray_linalg;
extern crate num_traits;

use ndarray::*;
use std::fs::*;
use std::io::Write;

use eom::traits::*;
use eom::*;

fn check_accuracy<Sc>(teo: Sc, init: Array1<f64>, fname: &str)
where
    Sc: Scheme<Scalar = f64, Dim = Ix1, Time = f64>,
{
    let acc = adaptor::accuracy(teo, init, 0.01, 1000, 12);
    let mut f = File::create(fname).unwrap();
    writeln!(&mut f, "dt,dev").unwrap();
    for &(dt, dev) in acc.iter() {
        writeln!(&mut f, "{:.08e},{:.08e}", dt, dev).unwrap();
    }
}

fn main() {
    let l63 = ode::Lorenz63::default();
    check_accuracy(
        explicit::Euler::new(l63, 1.0),
        arr1(&[1.0, 0.0, 0.0]),
        "euler.csv",
    );
    check_accuracy(
        explicit::Heun::new(l63, 1.0),
        arr1(&[1.0, 0.0, 0.0]),
        "heun.csv",
    );
    check_accuracy(
        explicit::RK4::new(l63, 1.0),
        arr1(&[1.0, 0.0, 0.0]),
        "rk4.csv",
    );
    check_accuracy(
        semi_implicit::DiagRK4::new(l63, 1.0),
        arr1(&[1.0, 0.0, 0.0]),
        "diag_rk4.csv",
    );
}
