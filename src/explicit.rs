//! Define explicit schemes

use ndarray::*;
use ndarray_linalg::*;
use super::traits::*;

macro_rules! def_explicit {
    ($method:ident, $constructor:ident) => {

#[derive(new)]
pub struct $method<F, Time> {
    f: F,
    dt: Time,
}

impl<F, Time: Copy> TimeStep<Time> for $method<F, Time> {
    fn get_dt(&self) -> Time {
        self.dt
    }
    fn set_dt(&mut self, dt: Time) {
        self.dt = dt;
    }
}

pub fn $constructor<F, Time>(f: F, dt: Time) -> $method<F, Time> {
    $method::new(f, dt)
}

}} // def_explicit

def_explicit!(Euler, euler);
def_explicit!(Heun, heun);
def_explicit!(RK4, rk4);

macro_rules! impl_time_evolution {
    ( $($mut_:tt), * ) => {

impl<'a, A, S, D, F> TimeEvolution<A, S, D> for &'a $($mut_),* Euler<F, A::Real>
    where A: Scalar,
          S: DataMut<Elem=A>,
          D: Dimension,
          for<'b> &'b $($mut_),* F: EOM<A, S, D>
{
    #[inline(always)]
    fn iterate(self, mut x: &mut ArrayBase<S, D>) -> &mut ArrayBase<S, D> {
        let x_ = x.to_owned();
        let fx = self.f.rhs(x);
        Zip::from(&mut *fx).and(&x_).apply(|vfx, vx| {
            *vfx = *vx + vfx.mul_real(self.dt);
        });
        fx
    }
}

impl<'a, A, S, D, F> TimeEvolution<A, S, D> for &'a $($mut_),* Heun<F, A::Real>
    where A: Scalar,
          S: DataMut<Elem=A>,
          D: Dimension,
          for<'b> &'b $($mut_),* F: EOM<A, S, D>
{
    #[inline(always)]
    fn iterate(self, mut x: &mut ArrayBase<S, D>) -> &mut ArrayBase<S, D> {
        let dt = self.dt;
        let dt_2 = self.dt * into_scalar(0.5);
        // calc
        let x_ = x.to_owned();
        let k1 = self.f.rhs(x);
        let k1_ = k1.to_owned();
        Zip::from(&mut *k1).and(&x_).apply(|k1, &x_|{
            *k1 = k1.mul_real(dt) + x_;
        });
        let k2 = self.f.rhs(k1);
        Zip::from(&mut *k2).and(&x_).and(&k1_).apply(|k2, &x_, &k1_| {
            *k2 = x_ + (k1_ + *k2).mul_real(dt_2);
        });
        k2
    }
}

impl<'a, A, S, D, F> TimeEvolution<A, S, D> for &'a $($mut_),* RK4<F, A::Real>
    where A: Scalar,
          S: DataMut<Elem=A>,
          D: Dimension,
          for<'b> &'b $($mut_),* F: EOM<A, S, D>
{
    #[inline(always)]
    fn iterate(self, mut x: &mut ArrayBase<S, D>) -> &mut ArrayBase<S, D> {
        let dt = self.dt;
        let dt_2 = self.dt * into_scalar(0.5);
        let dt_6 = self.dt / into_scalar(6.0);
        let x_ = x.to_owned();
        // k1
        let mut k1 = self.f.rhs(x);
        let k1_ = k1.to_owned();
        Zip::from(&mut *k1).and(&x_).apply(|k1, &x_| {
            *k1 = k1.mul_real(dt_2) + x_;
        });
        // k2
        let mut k2 = self.f.rhs(k1);
        let k2_ = k2.to_owned();
        Zip::from(&mut *k2).and(&x_).apply(|k2, &x_| {
            *k2 = x_ + k2.mul_real(dt_2);
        });
        // k3
        let mut k3 = self.f.rhs(k2);
        let k3_ = k3.to_owned();
        Zip::from(&mut *k3).and(&x_).apply(|k3, &x_| {
            *k3 = x_ + k3.mul_real(dt);
        });
        let mut k4 = self.f.rhs(k3);
        Zip::from(&mut *k4).and(&x_).and(&k1_).and(&k2_).and(&k3_).apply(|k4, &x_, &k1_, &k2_, &k3_| {
            *k4 = x_ + (k1_ + (k2_ + k3_).mul_real(into_scalar(2.0)) + *k4).mul_real(dt_6);
        });
        k4
    }
}

}} // impl_time_evolution!

impl_time_evolution!();
impl_time_evolution!(mut);
