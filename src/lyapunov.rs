use ndarray::*;
use ndarray_linalg::*;
use num_traits::{Float, One};

use traits::*;
use jacobian::*;

/// Calculate all Lyapunov exponents
///
/// This is an example usage of `Series` itertor, with which you can write more flexible procedure.
pub fn exponents<A, TEO>(teo: TEO, x: Array1<A>, alpha: A::Real, duration: usize) -> Array1<A::Real>
where
    A: Scalar,
    TEO: TimeEvolution<Scalar = A, Dim = Ix1> + TimeStep<Time = A::Real>,
{
    let n = teo.model_size();
    let dur = teo.get_dt() * into_scalar(duration as f64);
    Series::new(teo, x, alpha)
        .map(|(_x, _q, r)| r.diag().map(|x| x.abs().ln()))
        .skip(duration / 10)
        .take(duration)
        .fold(ArrayBase::zeros(n), |mut x, y| {
            azip!(mut x, y in { *x = *x + y/dur } );
            x
        })
}

/// An iterator for successive QR-decomposition in Lyapunov analysis
///
/// This is used both to calculate the Lyapunov exponents and covariant Lyapunov vector (CLV).
/// The `Item` of the iterator is `(x, Q, R)` where `x` is the state vector.
/// Be sure that each column of `Q` belongs to the tangent space at `x`,
/// and `R` is a map from the previous tangent space (i.e. at `F^{-1}(x)`) to the space spand by `Q`.
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
    type Item = (Array1<A>, Array2<A>, Array2<A>);

    fn next(&mut self) -> Option<Self::Item> {
        let q = self.teo
            .lin_approx(self.x.to_owned(), self.alpha)
            .apply_multi_inplace(&mut self.q);
        let (q, r) = q.qr_square_inplace().unwrap();
        self.teo.iterate(&mut self.x);
        Some((self.x.to_owned(), q.to_owned(), r))
    }
}

fn clv_backward<A: Scalar>(c: &Array2<A>, r: &Array2<A>) -> (Array2<A>, Array1<A::Real>) {
    let cd = r.solve_triangular(UPLO::Upper, ::ndarray_linalg::Diag::NonUnit, c)
        .expect("Failed to solve R");
    let (c, d) = normalize(cd, NormalizeAxis::Column);
    let f = Array::from_vec(d).mapv_into(|x| A::Real::one() / x);
    (c, f)
}

/// Calculate the Covariant Lyapunov Vectors at once
///
/// This function saves the time series of QR-decomposition, and consumes many memories.
pub fn vectors<A, TEO>(
    teo: TEO,
    x: Array1<A>,
    alpha: A::Real,
    duration: usize,
) -> Vec<(Array1<A>, Array2<A>, Array1<A::Real>)>
where
    A: Scalar,
    TEO: TimeEvolution<Scalar = A, Dim = Ix1> + Clone,
{
    let n = teo.model_size();
    let qr_series = Series::new(teo, x, alpha)
        .skip(duration / 10)
        .take(duration + duration / 10)
        .collect::<Vec<_>>();
    let clv_rev = qr_series
        .into_iter()
        .rev()
        .scan(Array::eye(n), |c, (x, q, r)| {
            let (c_now, f) = clv_backward(c, &r);
            let v = q.dot(&c_now);
            *c = c_now;
            Some((x, v, f))
        })
        .collect::<Vec<_>>();
    clv_rev.into_iter().skip(duration / 10).rev().collect()
}
