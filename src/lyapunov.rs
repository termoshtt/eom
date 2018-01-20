use ndarray::*;
use ndarray_linalg::*;
use num_traits::Float;

use traits::*;
use jacobian::*;

/// Calculate all Lyapunov exponents
pub fn exponents<A, TEO>(teo: TEO, x: Array1<A>, alpha: A::Real, duration: usize) -> Array1<A::Real>
where
    A: Scalar,
    TEO: TimeEvolution<Scalar = A, Dim = Ix1> + TimeStep<Time = A::Real>,
{
    let n = teo.model_size();
    let dur = teo.get_dt() * into_scalar(duration as f64);
    Series::new(teo, x, alpha)
        .map(|(_q, r)| r.diag().map(|x| x.abs().ln()))
        .skip(duration / 10)
        .take(duration)
        .fold(ArrayBase::zeros(n), |mut x, y| {
            azip!(mut x, y in { *x = *x + y/dur } );
            x
        })
}

pub struct Series<A, TEO>
where
    A: Scalar,
    TEO: TimeEvolution<Scalar = A, Dim = Ix1>,
{
    teo: TEO,
    x: Array1<A>,
    q: Array2<A>,
    alpha: A::Real,
}

impl<A, TEO> Series<A, TEO>
where
    A: Scalar,
    TEO: TimeEvolution<Scalar = A, Dim = Ix1>,
{
    pub fn new(teo: TEO, x: Array1<A>, alpha: A::Real) -> Self {
        let q = Array::eye(teo.model_size());
        Series { teo, x, q, alpha }
    }
}

impl<A, TEO> Iterator for Series<A, TEO>
where
    A: Scalar,
    TEO: TimeEvolution<Scalar = A, Dim = Ix1>,
{
    type Item = (Array2<A>, Array2<A>);

    fn next(&mut self) -> Option<Self::Item> {
        let q = self.teo
            .lin_approx(self.x.to_owned(), self.alpha)
            .apply_multi_inplace(&mut self.q);
        let (q, r) = q.qr_square_inplace().unwrap();
        self.teo.iterate(&mut self.x);
        Some((q.to_owned(), r))
    }
}
