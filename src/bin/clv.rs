
#[macro_use]
extern crate ndarray;
extern crate ndarray_odeint;
extern crate ndarray_linalg;
extern crate itertools;

use std::io::Write;
use ndarray::*;
use ndarray_linalg::prelude::*;
use ndarray_odeint::prelude::*;
use ndarray_odeint::lyapunov::*;
use itertools::iterate;

fn main() {
    let dt = 0.01;
    let eom = Lorenz63::default();
    let teo = explicit::rk4(eom, dt);
    let ts = iterate(rcarr1(&[1.0, 0.0, 0.0]), |y| teo.iterate(y.clone()));
    let duration = 100000;
    let qr_series = ts.scan(Array::eye(3), |st, x| {
            let j = teo.jacobian(x.clone(), 1e-7);
            let (q_next, r) = j.dot(st).qr().unwrap();
            let q = std::mem::replace(st, q_next);
            Some((q, r))
        })
        .skip(duration / 10)
        .take(duration)
        .collect::<Vec<_>>();
    let clv = qr_series.iter()
        .rev()
        .scan(Array::eye(3), |st, &(ref q, ref r)| {
            let cd = r.solve_upper(&*st).expect("Failed to solve R");
            let (c, d) = normalize(cd, NormalizeAxis::Column);
            let v = q.dot(&c);
            let f = Array::from_vec(d).mapv_into(|x| 1.0 / x);
            *st = c;
            Some((v, f))
        })
        .collect::<Vec<_>>();
    let mut l = Array::zeros(3);
    println!("v0v1, v0v2, v1v2");
    for (v, f) in clv.into_iter().rev() {
        let v0 = v.axis_iter(Axis(1)).nth(0).unwrap();
        let v1 = v.axis_iter(Axis(1)).nth(1).unwrap();
        let v2 = v.axis_iter(Axis(1)).nth(2).unwrap();
        println!("{}, {}, {}", v0.dot(&v1), v0.dot(&v2), v1.dot(&v2));
        l += &f.map(|x| x.abs().ln());
    }
    write!(&mut std::io::stderr(),
           "exponents = {:?}\n",
           l / (dt * duration as f64))
        .unwrap();
}
