//! Lyapunov Analysis

use ndarray::*;
use ndarray_linalg::*;
use itertools::iterate;
use std::mem::replace;

use super::traits::*;

pub use ndarray::linalg::Dot;

/// Jacobian operator using numerical-differentiation
pub struct Jacobian<'a, TEO>
    where TEO: 'a,
          &'a TEO: TimeEvolution<f64, OwnedRcRepr<f64>, Ix1>
{
    f: &'a TEO,
    x: RcArray1<f64>,
    fx: RcArray1<f64>,
    alpha: f64,
}

pub fn jacobian<'a, TEO>(f: &'a TEO, x: RcArray1<f64>, alpha: f64) -> Jacobian<'a, TEO>
    where TEO: 'a,
          &'a TEO: TimeEvolution<f64, OwnedRcRepr<f64>, Ix1>
{
    let fx = f.iterate(x.clone());
    Jacobian {
        f: f,
        x: x,
        fx: fx,
        alpha: alpha,
    }
}

impl<'a, S, TEO> Dot<ArrayBase<S, Ix1>> for Jacobian<'a, TEO>
    where &'a TEO: TimeEvolution<f64, OwnedRcRepr<f64>, Ix1>,
          S: Data<Elem = f64>
{
    type Output = RcArray1<f64>;
    fn dot(&self, dx: &ArrayBase<S, Ix1>) -> Self::Output {
        let nrm = self.x.norm_l2().max(dx.norm_l2());
        let n = self.alpha / nrm;
        let x = n * dx + &self.x;
        (self.f.iterate(x.into_shared()) - &self.fx) / n
    }
}

impl<'a, S, TEO> Dot<ArrayBase<S, Ix2>> for Jacobian<'a, TEO>
    where &'a TEO: TimeEvolution<f64, OwnedRcRepr<f64>, Ix1>,
          S: Data<Elem = f64>
{
    type Output = Array2<f64>;
    fn dot(&self, dxs: &ArrayBase<S, Ix2>) -> Self::Output {
        hstack(&dxs.axis_iter(Axis(1))
                   .map(|dx| self.dot(&dx))
                   .collect::<Vec<_>>())
            .unwrap()
    }
}

fn clv_backward(c: &Array2<f64>, r: &Array2<f64>) -> (Array2<f64>, Array1<f64>) {
    let cd = r.solve_triangular(UPLO::Upper, ::ndarray_linalg::Diag::NonUnit, c)
        .expect("Failed to solve R");
    let (c, d) = normalize(cd, NormalizeAxis::Column);
    let f = Array::from_vec(d).mapv_into(|x| 1.0 / x);
    (c, f)
}

/// Calculate all Lyapunov exponents
pub fn exponents<'a, TEO>(teo: &'a TEO,
                          x0: RcArray1<f64>,
                          alpha: f64,
                          duration: usize)
                          -> Array1<f64>
    where &'a TEO: TimeEvolution<f64, OwnedRcRepr<f64>, Ix1>,
          TEO: 'a + TimeStep<f64>
{
    let n = x0.len();
    let ts = iterate(x0, |y| teo.iterate(y.clone()));
    ts.scan(Array::eye(n), |q, x| {
        let (q_next, r): (_, Array2<f64>) = jacobian(teo, x.clone(), alpha).dot(q).qr().unwrap();
        *q = q_next;
        let d = r.diag().map(|x: &f64| x.abs().ln());
        Some(d)
    }).skip(duration / 10)
        .take(duration)
        .fold(Array::zeros(n), |x, y| x + y) / (teo.get_dt() * duration as f64)
}

/// Calculate Covariant Lyapunov Vector
///
/// **CAUTION**
/// This function consumes much memory since this saves matrices duraing the time evolution.
pub fn clv<'a, TEO>(teo: &'a TEO,
                    x0: RcArray1<f64>,
                    alpha: f64,
                    duration: usize)
                    -> Vec<(Array1<f64>, Array2<f64>, Array1<f64>)>
    where &'a TEO: TimeEvolution<f64, OwnedRcRepr<f64>, Ix1>,
          TEO: 'a
{
    let n = x0.len();
    let ts = iterate(x0, |y| teo.iterate(y.clone()));
    let qr_series = ts.scan(Array::eye(n), |q, x| {
        let (q_next, r) = jacobian(teo, x.clone(), alpha).dot(q).qr().unwrap();
        let q = replace(q, q_next);
        Some((x, q, r))
    }).skip(duration / 10)
        .take(duration + duration / 10)
        .collect::<Vec<_>>();
    let clv_rev = qr_series
        .into_iter()
        .rev()
        .scan(Array::eye(n), |c, (x, q, r)| {
            let (c_now, f) = clv_backward(c, &r);
            let v = q.dot(&c_now);
            *c = c_now;
            Some((x.into_owned(), v, f))
        })
        .collect::<Vec<_>>();
    clv_rev.into_iter().skip(duration / 10).rev().collect()
}
