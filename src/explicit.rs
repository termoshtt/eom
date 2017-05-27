//! Define explicit schemes

use ndarray::*;
use super::traits::*;

macro_rules! def_explicit {
    ($method:ident, $constructor:ident) => {

#[derive(new)]
pub struct $method<F> {
    f: F,
    dt: f64,
}

pub fn $constructor<F>(f: F, dt: f64) -> $method<F> {
    $method::new(f, dt)
}

impl<F> TimeStep for $method<F> {
    fn get_dt(&self) -> f64 {
        self.dt
    }
    fn set_dt(&mut self, dt: f64) {
        self.dt = dt;
    }
}

}} // def_explicit

def_explicit!(Euler, euler);
def_explicit!(Heun, heun);
def_explicit!(RK4, rk4);

macro_rules! impl_time_evolution {
    ( $($mut_:tt), * ) => {

impl<'a, A, D, F> TimeEvolution<A, OwnedRepr<A>, D> for &'a $($mut_),* Euler<F>
    where A: OdeScalar<f64>,
          D: Dimension,
          for<'b> &'b $($mut_),* F: EOM<A, OwnedRepr<A>, D>
{
    #[inline(always)]
    fn iterate(self, x: Array<A, D>) -> Array<A, D> {
        let x_ = x.to_owned();
        let mut fx = self.f.rhs(x);
        azip!(mut fx, x_ in { *fx = x_ + *fx * self.dt });
        fx
    }
}

impl<'a, A, D, F> TimeEvolution<A, OwnedRcRepr<A>, D> for &'a $($mut_),* Euler<F>
    where A: OdeScalar<f64>,
          D: Dimension,
          for<'b> &'b $($mut_),* F: EOM<A, OwnedRcRepr<A>, D>
{
    #[inline(always)]
    fn iterate(self, x: RcArray<A, D>) -> RcArray<A, D> {
        let x_ = x.to_owned();
        let mut fx = self.f.rhs(x);
        azip!(mut fx, x_ in { *fx = x_ + *fx * self.dt });
        fx
    }
}

impl<'a, A, D, F> TimeEvolution<A, ViewRepr<&'a mut A>, D> for &'a $($mut_),* Euler<F>
    where A: OdeScalar<f64>,
          D: Dimension,
          for<'b, 'c> &'b $($mut_),* F: EOM<A, ViewRepr<&'c mut A>, D>
{
    #[inline(always)]
    fn iterate(self, x: ArrayViewMut<A, D>) -> ArrayViewMut<A, D> {
        let x_ = x.to_owned();
        let mut fx = self.f.rhs(x);
        azip!(mut fx, x_ in { *fx = x_ + *fx * self.dt });
        fx
    }
}

impl<'a, A, D, F> TimeEvolution<A, OwnedRepr<A>, D> for &'a $($mut_),* Heun<F>
    where A: OdeScalar<f64>,
          D: Dimension,
          for<'b> &'b $($mut_),* F: EOM<A, OwnedRepr<A>, D>
{
    #[inline(always)]
    fn iterate(self, x: Array<A, D>) -> Array<A, D> {
        let dt = self.dt;
        let dt_2 = self.dt * 0.5;
        let x_ = x.to_owned();
        let mut k1 = self.f.rhs(x);
        let k1_ = k1.to_owned();
        azip!(mut k1, x_ in { *k1 = *k1 * dt + x_ });
        let mut k2 = self.f.rhs(k1);
        azip!(mut k2, x_, k1_ in { *k2 = x_ + (k1_ + *k2) * dt_2 });
        k2
    }
}

impl<'a, A, D, F> TimeEvolution<A, OwnedRcRepr<A>, D> for &'a $($mut_),* Heun<F>
    where A: OdeScalar<f64>,
          D: Dimension,
          for<'b> &'b $($mut_),* F: EOM<A, OwnedRcRepr<A>, D>
{
    #[inline(always)]
    fn iterate(self, x: RcArray<A, D>) -> RcArray<A, D> {
        let dt = self.dt;
        let dt_2 = self.dt * 0.5;
        let x_ = x.to_owned();
        let mut k1 = self.f.rhs(x);
        let k1_ = k1.to_owned();
        azip!(mut k1, x_ in { *k1 = *k1 * dt + x_ });
        let mut k2 = self.f.rhs(k1);
        azip!(mut k2, x_, k1_ in { *k2 = x_ + (k1_ + *k2) * dt_2 });
        k2
    }
}

impl<'a, A, D, F> TimeEvolution<A, ViewRepr<&'a mut A>, D> for &'a $($mut_),* Heun<F>
    where A: OdeScalar<f64>,
          D: Dimension,
          for<'b, 'c> &'b $($mut_),* F: EOM<A, ViewRepr<&'c mut A>, D>
{
    #[inline(always)]
    fn iterate(self, x: ArrayViewMut<A, D>) -> ArrayViewMut<A, D> {
        let dt = self.dt;
        let dt_2 = self.dt * 0.5;
        let x_ = x.to_owned();
        let mut k1 = self.f.rhs(x);
        let k1_ = k1.to_owned();
        azip!(mut k1, x_ in { *k1 = *k1 * dt + x_ });
        let mut k2 = self.f.rhs(k1);
        azip!(mut k2, x_, k1_ in { *k2 = x_ + (k1_ + *k2) * dt_2 });
        k2
    }
}

impl<'a, A, D, F> TimeEvolution<A, OwnedRepr<A>, D> for &'a $($mut_),* RK4<F>
    where A: OdeScalar<f64>,
          D: Dimension,
          for<'b> &'b $($mut_),* F: EOM<A, OwnedRepr<A>, D>
{
    #[inline(always)]
    fn iterate(self, x: Array<A, D>) -> Array<A, D> {
        let dt = self.dt;
        let dt_2 = self.dt * 0.5;
        let dt_6 = self.dt / 6.0;
        let x_ = x.to_owned();
        // k1
        let mut k1 = self.f.rhs(x);
        let k1_ = k1.to_owned();
        azip!(mut k1, x_ in { *k1 = *k1 * dt_2 + x_ });
        // k2
        let mut k2 = self.f.rhs(k1);
        let k2_ = k2.to_owned();
        azip!(mut k2, x_ in { *k2 = x_ + *k2 * dt_2 });
        // k3
        let mut k3 = self.f.rhs(k2);
        let k3_ = k3.to_owned();
        azip!(mut k3, x_ in { *k3 = x_ + *k3 * dt });
        let mut k4 = self.f.rhs(k3);
        azip!(mut k4, x_, k1_, k2_, k3_ in {
            *k4 = x_ + (k1_ + (k2_ + k3_) * 2.0 + *k4) * dt_6
        });
        k4
    }
}

impl<'a, A, D, F> TimeEvolution<A, OwnedRcRepr<A>, D> for &'a $($mut_),* RK4<F>
    where A: OdeScalar<f64>,
          D: Dimension,
          for<'b> &'b $($mut_),* F: EOM<A, OwnedRcRepr<A>, D>
{
    #[inline(always)]
    fn iterate(self, x: RcArray<A, D>) -> RcArray<A, D> {
        let dt = self.dt;
        let dt_2 = self.dt * 0.5;
        let dt_6 = self.dt / 6.0;
        let x_ = x.to_owned();
        // k1
        let mut k1 = self.f.rhs(x);
        let k1_ = k1.to_owned();
        azip!(mut k1, x_ in { *k1 = *k1 * dt_2 + x_ });
        // k2
        let mut k2 = self.f.rhs(k1);
        let k2_ = k2.to_owned();
        azip!(mut k2, x_ in { *k2 = x_ + *k2 * dt_2 });
        // k3
        let mut k3 = self.f.rhs(k2);
        let k3_ = k3.to_owned();
        azip!(mut k3, x_ in { *k3 = x_ + *k3 * dt });
        let mut k4 = self.f.rhs(k3);
        azip!(mut k4, x_, k1_, k2_, k3_ in {
            *k4 = x_ + (k1_ + (k2_ + k3_) * 2.0 + *k4) * dt_6
        });
        k4
    }
}

impl<'a, A, D, F> TimeEvolution<A, ViewRepr<&'a mut A>, D> for &'a $($mut_),* RK4<F>
    where A: OdeScalar<f64>,
          D: Dimension,
          for<'b, 'c> &'b $($mut_),* F: EOM<A, ViewRepr<&'c mut A>, D>
{
    #[inline(always)]
    fn iterate(self, x: ArrayViewMut<A, D>) -> ArrayViewMut<A, D> {
        let dt = self.dt;
        let dt_2 = self.dt * 0.5;
        let dt_6 = self.dt / 6.0;
        let x_ = x.to_owned();
        // k1
        let mut k1 = self.f.rhs(x);
        let k1_ = k1.to_owned();
        azip!(mut k1, x_ in { *k1 = *k1 * dt_2 + x_ });
        // k2
        let mut k2 = self.f.rhs(k1);
        let k2_ = k2.to_owned();
        azip!(mut k2, x_ in { *k2 = x_ + *k2 * dt_2 });
        // k3
        let mut k3 = self.f.rhs(k2);
        let k3_ = k3.to_owned();
        azip!(mut k3, x_ in { *k3 = x_ + *k3 * dt });
        let mut k4 = self.f.rhs(k3);
        azip!(mut k4, x_, k1_, k2_, k3_ in {
            *k4 = x_ + (k1_ + (k2_ + k3_) * 2.0 + *k4) * dt_6
        });
        k4
    }
}

}} // impl_time_evolution!

impl_time_evolution!();
impl_time_evolution!(mut);
