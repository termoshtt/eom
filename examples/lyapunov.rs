//! Lyapunov Analysis example

extern crate ndarray;
extern crate eom;
extern crate ndarray_linalg;
extern crate num_traits;

use ndarray::*;
use ndarray_linalg::*;
use eom::*;
use num_traits::Float;

/// Calculate all Lyapunov exponents
pub fn exponents<A, S, TEO>(teo: TEO,
                            x0: ArrayBase<S, Ix1>,
                            alpha: A::Real,
                            duration: usize)
                            -> Array1<A>
    where A: RealScalar,
          S: DataMut<Elem = A> + DataClone,
          TEO: TimeEvolution<Scalar = A, Dim = Ix1> + TimeStep<Time = A>
{
    let n = x0.len();
    let dur = teo.get_dt() * into_scalar(duration as f64);
    let ts = time_series(x0, &teo);
    ts.scan(Array::eye(n), |q, x| {
        let q = jacobian(&teo, x.clone(), alpha).op_multi_mut(q);
        let (q_next, r) = q.qr().unwrap();
        *q = q_next;
        let d: Array1<A> = r.diag().map(|x| AssociatedReal::inject(x.abs().ln()));
        Some(d)
    }).skip(duration / 10)
        .take(duration)
        .fold(ArrayBase::zeros(n), |mut x, y| {
            azip!(mut x, y in { *x = *x + y/dur } );
            x
        })
}

fn main() {
    let dt = 0.01;
    let eom = model::Lorenz63::default();
    let teo = explicit::rk4(eom, dt);
    let l = exponents(teo, arr1(&[1.0, 0.0, 0.0]), 1e-7, 100000);
    println!("Lyapunov Exponents = {:?}", l);
    assert_close_l2!(&l, &arr1(&[0.906, 0.0, -14.572]), 1e-2);
    // value from http://sprott.physics.wisc.edu/chaos/lorenzle.htm
}
