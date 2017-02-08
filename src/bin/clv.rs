
extern crate ndarray;
extern crate ndarray_odeint;
extern crate ndarray_linalg;
extern crate itertools;

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
            *st = c;
            Some((v, d))
        })
        .collect::<Vec<_>>();
    let mut l = Array::zeros(3);
    for (_, d) in clv.into_iter().rev() {
        let mut f = Array::from_vec(d);
        f.map_inplace(|x| *x = x.abs().ln());
        l -= &f;
    }
    println!("{:?}", l / (dt * duration as f64));
}
