
use std::marker::PhantomData;
use ndarray::{RcArray, Dimension};
use super::traits::{EOM, TimeEvolution, OdeScalar};

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

impl<'a, A, D, F> TimeEvolution<A, D> for &'a Explicit<F, markers::EulerMarker>
    where A: OdeScalar<f64>,
          D: Dimension,
          &'a F: EOM<A, D>
{
    #[inline(always)]
    fn iterate(self, x: RcArray<A, D>) -> RcArray<A, D> {
        let fx = self.f.rhs(x.clone());
        x + fx * self.dt
    }
    fn get_dt(&self) -> f64 {
        self.dt
    }
}

impl<'a, A, D, F> TimeEvolution<A, D> for &'a mut Explicit<F, markers::EulerMarker>
    where A: OdeScalar<f64>,
          D: Dimension,
          &'a mut F: EOM<A, D>
{
    #[inline(always)]
    fn iterate(self, x: RcArray<A, D>) -> RcArray<A, D> {
        let fx = self.f.rhs(x.clone());
        x + fx * self.dt
    }
    fn get_dt(&self) -> f64 {
        self.dt
    }
}

impl<'a, A, D, F> TimeEvolution<A, D> for &'a Explicit<F, markers::HeunMarker>
    where A: OdeScalar<f64>,
          D: Dimension,
          &'a F: EOM<A, D>
{
    #[inline(always)]
    fn iterate(self, x: RcArray<A, D>) -> RcArray<A, D> {
        let k1 = self.f.rhs(x.clone()) * self.dt;
        let k2 = self.f.rhs(x.clone() + k1.clone()) * self.dt;
        x + (k1 + k2) * 0.5
    }
    fn get_dt(&self) -> f64 {
        self.dt
    }
}

impl<'a, A, D, F> TimeEvolution<A, D> for &'a mut Explicit<F, markers::HeunMarker>
    where A: OdeScalar<f64>,
          D: Dimension,
          for<'b> &'b mut F: EOM<A, D>
{
    #[inline(always)]
    fn iterate(self, x: RcArray<A, D>) -> RcArray<A, D> {
        let k1 = self.f.rhs(x.clone()) * self.dt;
        let k2 = self.f.rhs(x.clone() + k1.clone()) * self.dt;
        x + (k1 + k2) * 0.5
    }
    fn get_dt(&self) -> f64 {
        self.dt
    }
}

impl<'a, A, D, F> TimeEvolution<A, D> for &'a Explicit<F, markers::RK4Marker>
    where A: OdeScalar<f64>,
          D: Dimension,
          &'a F: EOM<A, D>
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
    fn get_dt(&self) -> f64 {
        self.dt
    }
}

impl<'a, A, D, F> TimeEvolution<A, D> for &'a mut Explicit<F, markers::RK4Marker>
    where A: OdeScalar<f64>,
          D: Dimension,
          for<'b> &'b mut F: EOM<A, D>
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
    fn get_dt(&self) -> f64 {
        self.dt
    }
}

pub fn euler<F>(f: F, dt: f64) -> Explicit<F, markers::EulerMarker> {
    Explicit::new(f, dt)
}

pub fn heun<F>(f: F, dt: f64) -> Explicit<F, markers::HeunMarker> {
    Explicit::new(f, dt)
}

pub fn rk4<F>(f: F, dt: f64) -> Explicit<F, markers::RK4Marker> {
    Explicit::new(f, dt)
}
