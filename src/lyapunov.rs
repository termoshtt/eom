use ndarray::*;
use ndarray_linalg::*;
use num_traits::Float;

use traits::*;
use jacobian::*;
use adaptor::time_series;

/// Calculate all Lyapunov exponents
pub fn exponents<A, S, TEO>(
    mut teo: TEO,
    x0: ArrayBase<S, Ix1>,
    alpha: A::Real,
    duration: usize,
) -> Array1<A>
where
    A: RealScalar,
    S: DataMut<Elem = A> + DataClone,
    TEO: TimeEvolution<Scalar = A, Dim = Ix1> + TimeStep<Time = A>,
{
    let n = x0.len();
    let dur = teo.get_dt() * into_scalar(duration as f64);
    let mut teo2 = teo.clone();
    let ts = time_series(x0, &mut teo);
    ts.scan(Array::eye(n), |q, x| {
        teo2.lin_approx(x.to_owned(), alpha).apply_multi_inplace(q);
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
