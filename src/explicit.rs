
use std::marker::PhantomData;
use ndarray::{RcArray, Dimension};
use super::traits::{EOM, TimeEvolution, OdeScalar};

pub mod markers {
    pub struct EulerMarker {}
    pub struct HeunMarker {}
    pub struct RK4Marker {}
}

pub struct Explicit<A: OdeScalar<f64>, F: EOM<A, D>, D: Dimension, Marker> {
    f: F,
    dt: f64,
    phantom: PhantomData<(A, D, Marker)>,
}

impl<A: OdeScalar<f64>, F: EOM<A, D>, D: Dimension, Marker> Explicit<A, F, D, Marker> {
    pub fn new(f: F, dt: f64) -> Self {
        Explicit {
            f: f,
            dt: dt,
            phantom: PhantomData,
        }
    }
}

impl<A: OdeScalar<f64>, F: EOM<A, D>, D: Dimension> TimeEvolution<A, D>
    for Explicit<A, F, D, markers::EulerMarker> {
    #[inline(always)]
    fn iterate(&self, x: RcArray<A, D>) -> RcArray<A, D> {
        let fx = self.f.rhs(x.clone());
        x + fx * self.dt
    }
    fn get_dt(&self) -> f64 {
        self.dt
    }
}

impl<A: OdeScalar<f64>, F: EOM<A, D>, D: Dimension> TimeEvolution<A, D>
    for Explicit<A, F, D, markers::HeunMarker> {
    #[inline(always)]
    fn iterate(&self, x: RcArray<A, D>) -> RcArray<A, D> {
        let k1 = self.f.rhs(x.clone()) * self.dt;
        let k2 = self.f.rhs(x.clone() + k1.clone()) * self.dt;
        x + (k1 + k2) * 0.5
    }
    fn get_dt(&self) -> f64 {
        self.dt
    }
}

impl<A: OdeScalar<f64>, F: EOM<A, D>, D: Dimension> TimeEvolution<A, D>
    for Explicit<A, F, D, markers::RK4Marker> {
    #[inline(always)]
    fn iterate(&self, x: RcArray<A, D>) -> RcArray<A, D> {
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

pub fn euler<A, F, D>(f: F, dt: f64) -> Explicit<A, F, D, markers::EulerMarker>
    where A: OdeScalar<f64>,
          F: EOM<A, D>,
          D: Dimension
{
    Explicit::new(f, dt)
}

pub fn heun<A, F, D>(f: F, dt: f64) -> Explicit<A, F, D, markers::HeunMarker>
    where A: OdeScalar<f64>,
          F: EOM<A, D>,
          D: Dimension
{
    Explicit::new(f, dt)
}

pub fn rk4<A, F, D>(f: F, dt: f64) -> Explicit<A, F, D, markers::RK4Marker>
    where A: OdeScalar<f64>,
          F: EOM<A, D>,
          D: Dimension
{
    Explicit::new(f, dt)
}
