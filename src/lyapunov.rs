//! Lyapunov Analysis for time-evolution operators

use ndarray::*;
use ndarray_linalg::*;
use num_traits::{Float, FromPrimitive, One};

use crate::traits::*;

/// Jacobian operator using numerical-differentiation
pub struct Jacobian<'jac, A, D, TEO>
where
    A: Scalar + Lapack,
    D: Dimension,
    TEO: 'jac + TimeEvolution<Scalar = A, Dim = D>,
{
    f: &'jac mut TEO,
    x: Array<A, D>,
    fx: Array<A, D>,
    alpha: A::Real,
}

pub trait LinearApprox<A, D, TEO>
where
    A: Scalar + Lapack,
    D: Dimension,
    TEO: TimeEvolution<Scalar = A, Dim = D>,
{
    fn lin_approx<'jac>(
        &'jac mut self,
        x: Array<A, D>,
        alpha: A::Real,
    ) -> Jacobian<'jac, A, D, TEO>
    where
        TEO: 'jac;
}

impl<A, D, TEO> LinearApprox<A, D, TEO> for TEO
where
    A: Scalar + Lapack,
    D: Dimension,
    TEO: TimeEvolution<Scalar = A, Dim = D>,
{
    fn lin_approx<'jac>(&'jac mut self, x: Array<A, D>, alpha: A::Real) -> Jacobian<'jac, A, D, TEO>
    where
        TEO: 'jac,
    {
        Jacobian::new(self, x, alpha)
    }
}

impl<'jac, A, D, TEO> Jacobian<'jac, A, D, TEO>
where
    A: Scalar + Lapack,
    D: Dimension,
    TEO: TimeEvolution<Scalar = A, Dim = D>,
{
    pub fn new(f: &'jac mut TEO, x: Array<A, D>, alpha: A::Real) -> Jacobian<'jac, A, D, TEO>
    where
        TEO: 'jac,
    {
        let mut fx = x.clone();
        f.iterate(&mut fx);
        Jacobian { f, x, fx, alpha }
    }

    pub fn apply(&mut self, mut dx: Array<A, D>) -> Array<A, D> {
        self.apply_inplace(&mut dx);
        dx
    }

    pub fn apply_inplace<'a, S>(&mut self, dx: &'a mut ArrayBase<S, D>) -> &'a mut ArrayBase<S, D>
    where
        S: DataMut<Elem = A>,
    {
        let dx_nrm = dx.norm_l2().max(self.alpha);
        let n = self.alpha / dx_nrm;
        Zip::from(&mut *dx).and(&self.x).apply(|dx, &x| {
            *dx = x + dx.mul_real(n);
        });
        let x_dx = self.f.iterate(dx);
        Zip::from(&mut *x_dx).and(&self.fx).apply(|x_dx, &fx| {
            *x_dx = (*x_dx - fx).div_real(n);
        });
        x_dx
    }

    pub fn apply_multi(&mut self, mut a: Array<A, D::Larger>) -> Array<A, D::Larger>
    where
        D::Larger: RemoveAxis + Dimension<Smaller = D>,
    {
        self.apply_multi_inplace(&mut a);
        a
    }

    pub fn apply_multi_inplace<'a, S>(
        &mut self,
        a: &'a mut ArrayBase<S, D::Larger>,
    ) -> &'a mut ArrayBase<S, D::Larger>
    where
        S: DataMut<Elem = A>,
        D::Larger: RemoveAxis + Dimension<Smaller = D>,
    {
        let n = a.ndim();
        for mut col in a.axis_iter_mut(Axis(n - 1)) {
            self.apply_inplace(&mut col);
        }
        a
    }
}

/// Calculate all Lyapunov exponents
///
/// This is an example usage of `Series` itertor, with which you can write more flexible procedure.
pub fn exponents<A, TEO>(teo: TEO, x: Array1<A>, alpha: A::Real, duration: usize) -> Array1<A::Real>
where
    A: Scalar + Lapack,
    TEO: TimeEvolution<Scalar = A, Dim = Ix1> + TimeStep<Time = A::Real>,
{
    let n = teo.model_size();
    let dur = teo.get_dt() * TEO::Time::from_usize(duration).unwrap();
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
    A: Scalar + Lapack,
    TEO: TimeEvolution<Scalar = A, Dim = Ix1>,
{
    teo: TEO,
    x: Array1<A>,
    q: Array2<A>,
    alpha: A::Real,
}

impl<A, TEO> Series<A, TEO>
where
    A: Scalar + Lapack,
    TEO: TimeEvolution<Scalar = A, Dim = Ix1>,
{
    pub fn new(teo: TEO, x: Array1<A>, alpha: A::Real) -> Self {
        let q = Array::eye(teo.model_size());
        Series { teo, x, q, alpha }
    }
}

impl<A, TEO> Iterator for Series<A, TEO>
where
    A: Scalar + Lapack,
    TEO: TimeEvolution<Scalar = A, Dim = Ix1>,
{
    type Item = (Array1<A>, Array2<A>, Array2<A>);

    fn next(&mut self) -> Option<Self::Item> {
        let q = self
            .teo
            .lin_approx(self.x.to_owned(), self.alpha)
            .apply_multi_inplace(&mut self.q);
        let (q, r) = q.qr_square_inplace().unwrap();
        self.teo.iterate(&mut self.x);
        Some((self.x.to_owned(), q.to_owned(), r))
    }
}

fn clv_backward<A: Scalar + Lapack>(c: &Array2<A>, r: &Array2<A>) -> (Array2<A>, Array1<A::Real>) {
    let cd = r
        .solve_triangular(UPLO::Upper, ::ndarray_linalg::Diag::NonUnit, c)
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
    A: Scalar + Lapack,
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
