
use std::marker::PhantomData;
use ndarray::*;
use super::traits::*;

pub mod markers {
    pub struct EulerMarker {}
    pub struct HeunMarker {}
    pub struct RK4Marker {}
}

pub struct Explicit<F, Marker> {
    f: F,
    dt: f64,
    phantom: PhantomData<Marker>,
}

impl<F, Marker> Explicit<F, Marker> {
    pub fn new(f: F, dt: f64) -> Self {
        Explicit {
            f: f,
            dt: dt,
            phantom: PhantomData,
        }
    }
}

impl<F, Marker> TimeStep for Explicit<F, Marker> {
    fn get_dt(&self) -> f64 {
        self.dt
    }
    fn set_dt(&mut self, dt: f64) {
        self.dt = dt;
    }
}

macro_rules! impl_time_evolution {
    ( $($mut_:tt), * ) => {

impl<'a, A, D, F> TimeEvolution<A, OwnedRcRepr<A>, D> for &'a $($mut_),* Explicit<F, markers::EulerMarker>
    where A: OdeScalar<f64>,
          D: Dimension,
          for<'b> &'b $($mut_),* F: EOM<A, OwnedRcRepr<A>, D>
{
    #[inline(always)]
    fn iterate(self, x: RcArray<A, D>) -> RcArray<A, D> {
        let fx = self.f.rhs(x.clone());
        x + fx * self.dt
    }
}

impl<'a, 'b, A, D, F> TimeEvolution<A, ViewRepr<&'b mut A>, D> for &'a $($mut_),* Explicit<F, markers::EulerMarker>
    where A: OdeScalar<f64>,
          D: Dimension,
          for<'c> &'c $($mut_),* F: EOM<A, ViewRepr<&'b mut A>, D>
{
    #[inline(always)]
    fn iterate(self, x: ArrayViewMut<'b ,A, D>) -> ArrayViewMut<'b, A, D> {
        let x_ = x.to_owned();
        let mut fx = self.f.rhs(x);
        azip!(mut fx, x_ in { *fx = x_ + *fx * self.dt });
        fx
    }
}

impl<'a, A, D, F> TimeEvolution<A, OwnedRcRepr<A>, D> for &'a $($mut_),* Explicit<F, markers::HeunMarker>
    where A: OdeScalar<f64>,
          D: Dimension,
          for<'b> &'b $($mut_),* F: EOM<A, OwnedRcRepr<A>, D>
{
    #[inline(always)]
    fn iterate(self, x: RcArray<A, D>) -> RcArray<A, D> {
        let k1 = self.f.rhs(x.clone()) * self.dt;
        let k2 = self.f.rhs(x.clone() + k1.clone()) * self.dt;
        x + (k1 + k2) * 0.5
    }
}

impl<'a, 'b, A, D, F> TimeEvolution<A, ViewRepr<&'b mut A>, D> for &'a $($mut_),* Explicit<F, markers::HeunMarker>
    where A: OdeScalar<f64>,
          D: Dimension,
          for<'c> &'c $($mut_),* F: EOM<A, ViewRepr<&'b mut A>, D>
{
    #[inline(always)]
    fn iterate(self, x: ArrayViewMut<'b ,A, D>) -> ArrayViewMut<'b, A, D> {
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

impl<'a, A, D, F> TimeEvolution<A, OwnedRcRepr<A>, D> for &'a $($mut_),* Explicit<F, markers::RK4Marker>
    where A: OdeScalar<f64>,
          D: Dimension,
          for<'b> &'b $($mut_),* F: EOM<A, OwnedRcRepr<A>, D>
{
    #[inline(always)]
    fn iterate(self, x: RcArray<A, D>) -> RcArray<A, D> {
        let dt_2 = 0.5 * self.dt;
        let dt_6 = self.dt / 6.0;
        let k1 = self.f.rhs(x.clone());
        let l1 = k1.clone() * dt_2 + &x;
        let k2 = self.f.rhs(l1);
        let l2 = k2.clone() * dt_2 + &x;
        let k3 = self.f.rhs(l2);
        let l3 = k3.clone() * self.dt + &x;
        let k4 = self.f.rhs(l3);
        x + (k1 + (k2 + k3) * 2.0 + k4) * dt_6
    }
}

impl<'a, 'b, A, D, F> TimeEvolution<A, ViewRepr<&'b mut A>, D> for &'a $($mut_),* Explicit<F, markers::RK4Marker>
    where A: OdeScalar<f64>,
          D: Dimension,
          for<'c> &'c $($mut_),* F: EOM<A, ViewRepr<&'b mut A>, D>
{
    #[inline(always)]
    fn iterate(self, x: ArrayViewMut<'b ,A, D>) -> ArrayViewMut<'b, A, D> {
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

pub fn euler<F>(f: F, dt: f64) -> Explicit<F, markers::EulerMarker> {
    Explicit::new(f, dt)
}

pub fn heun<F>(f: F, dt: f64) -> Explicit<F, markers::HeunMarker> {
    Explicit::new(f, dt)
}

pub fn rk4<F>(f: F, dt: f64) -> Explicit<F, markers::RK4Marker> {
    Explicit::new(f, dt)
}
