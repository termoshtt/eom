
use std::marker::PhantomData;
use ndarray::prelude::*;
use super::traits::{EOM, TimeEvolution};

pub mod markers {
    pub struct EulerMarker {}
    pub struct HeunMarker {}
    pub struct RK4Marker {}
}

pub struct Explicit<F: EOM<D>, D: Dimension, Marker> {
    f: F,
    dt: f64,
    phantom_dim: PhantomData<D>,
    phantom_marker: PhantomData<Marker>,
}

impl<F: EOM<D>, D: Dimension, Marker> Explicit<F, D, Marker> {
    pub fn new(f: F, dt: f64) -> Self {
        Explicit {
            f: f,
            dt: dt,
            phantom_dim: PhantomData,
            phantom_marker: PhantomData,
        }
    }
}

impl<F: EOM<D>, D: Dimension> TimeEvolution<D> for Explicit<F, D, markers::EulerMarker> {
    #[inline(always)]
    fn iterate(&self, x: RcArray<f64, D>) -> RcArray<f64, D> {
        let fx = self.f.rhs(x.clone());
        x + fx * self.dt
    }
}

impl<F: EOM<D>, D: Dimension> TimeEvolution<D> for Explicit<F, D, markers::HeunMarker> {
    #[inline(always)]
    fn iterate(&self, x: RcArray<f64, D>) -> RcArray<f64, D> {
        let k1 = self.dt * self.f.rhs(x.clone());
        let k2 = self.dt * self.f.rhs(x.clone() + k1.clone());
        x + 0.5 * (k1 + k2)
    }
}

impl<F: EOM<D>, D: Dimension> TimeEvolution<D> for Explicit<F, D, markers::RK4Marker> {
    #[inline(always)]
    fn iterate(&self, x: RcArray<f64, D>) -> RcArray<f64, D> {
        let mut l = x.clone();
        l = self.f.rhs(l);
        let k1 = l.clone();
        l = (0.5 * self.dt) * l + &x;
        l = self.f.rhs(l);
        let k2 = l.clone();
        l = (0.5 * self.dt) * l + &x;
        l = self.f.rhs(l);
        let k3 = l.clone();
        l = self.dt * l + &x;
        l = self.f.rhs(l);
        x + (self.dt / 6.0) * (k1 + 2.0 * (k2 + k3) + l)
    }
}

pub fn euler<F: EOM<D>, D: Dimension>(f: F, dt: f64) -> Explicit<F, D, markers::EulerMarker> {
    Explicit::new(f, dt)
}

pub fn heun<F: EOM<D>, D: Dimension>(f: F, dt: f64) -> Explicit<F, D, markers::HeunMarker> {
    Explicit::new(f, dt)
}

pub fn rk4<F: EOM<D>, D: Dimension>(f: F, dt: f64) -> Explicit<F, D, markers::RK4Marker> {
    Explicit::new(f, dt)
}
