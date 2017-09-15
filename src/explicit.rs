//! Define explicit schemes

use ndarray::*;
use ndarray_linalg::*;
use super::traits::*;

use std::marker::PhantomData;

macro_rules! def_explicit {
    ($method:ident, $constructor:ident) => {

pub struct $method<A, D, F> 
where A: Scalar,
      D: Dimension,
      F: Explicit<D, Scalar = A> + TimeStep<Time = A::Real>
{
    f: F,
    dt: A::Real,
    phantom: PhantomData<D>
}

impl<A, D, F> ModelSize<D> for $method<A, D, F>
where A: Scalar,
      D: Dimension,
      F: Explicit<D, Scalar = A> + TimeStep<Time = A::Real> + ModelSize<D>
{
    fn model_size(&self) -> D::Pattern {
        self.f.model_size()
    }
}

impl<A, D, F> TimeStep for $method<A, D, F>
where A: Scalar,
      D: Dimension,
      F: Explicit<D, Scalar = A> + TimeStep<Time = A::Real>
{
    type Time = A::Real;
    fn get_dt(&self) -> Self::Time {
        self.dt
    }
    fn set_dt(&mut self, dt: Self::Time) {
        self.dt = dt;
    }
}

pub fn $constructor<A, D, F>(f: F, dt: A::Real) -> $method<A, D, F>
where A: Scalar,
      D: Dimension,
      F: Explicit<D, Scalar = A> + TimeStep<Time = A::Real>
{
    $method { f: f, dt: dt, phantom: PhantomData }
}

}} // def_explicit

def_explicit!(Euler, euler);
def_explicit!(Heun, heun);
def_explicit!(RK4, rk4);

pub struct EulerBuffer<A, D> {
    x: Array<A, D>,
}

impl<A, D, F> WithBuffer for Euler<A, D, F>
    where A: Scalar,
          D: Dimension,
          F: Explicit<D, Scalar = A> + TimeStep<Time = A::Real> + ModelSize<D>
{
    type Buffer = EulerBuffer<A, D>;

    fn new_buffer(&self) -> Self::Buffer {
        EulerBuffer { x: Array::zeros(self.f.model_size()) }
    }
}

impl<A, D, F> TimeEvolution<D> for Euler<A, D, F>
    where A: Scalar,
          D: Dimension,
          F: Explicit<D, Scalar = A> + TimeStep<Time = A::Real> + ModelSize<D>
{
    type Scalar = A;

    fn iterate<'a, S>(&self,
                      mut x: &'a mut ArrayBase<S, D>,
                      mut buf: &mut Self::Buffer)
                      -> &'a mut ArrayBase<S, D>
        where S: DataMut<Elem = A>
    {
        buf.x.zip_mut_with(x, |buf, x| *buf = *x);
        let mut fx = self.f.rhs(x);
        Zip::from(&mut *fx)
            .and(&buf.x)
            .apply(|vfx, vx| { *vfx = *vx + vfx.mul_real(self.dt); });
        fx
    }
}

pub struct HeunBuffer<A, D> {
    x: Array<A, D>,
    k1: Array<A, D>,
}

impl<A, D, F> WithBuffer for Heun<A, D, F>
    where A: Scalar,
          D: Dimension,
          F: Explicit<D, Scalar = A> + TimeStep<Time = A::Real> + ModelSize<D>
{
    type Buffer = HeunBuffer<A, D>;

    fn new_buffer(&self) -> Self::Buffer {
        HeunBuffer {
            x: Array::zeros(self.f.model_size()),
            k1: Array::zeros(self.f.model_size()),
        }
    }
}

impl<A, D, F> TimeEvolution<D> for Heun<A, D, F>
    where A: Scalar,
          D: Dimension,
          F: Explicit<D, Scalar = A> + TimeStep<Time = A::Real> + ModelSize<D>
{
    type Scalar = A;

    fn iterate<'a, S>(&self,
                      mut x: &'a mut ArrayBase<S, D>,
                      mut buf: &mut HeunBuffer<A, D>)
                      -> &'a mut ArrayBase<S, D>
        where S: DataMut<Elem = A>
    {
        let dt = self.dt;
        let dt_2 = self.dt * into_scalar(0.5);
        // calc
        buf.x.zip_mut_with(x, |buf, x| *buf = *x);
        let k1 = self.f.rhs(x);
        buf.k1.zip_mut_with(k1, |buf, k1| *buf = *k1);
        Zip::from(&mut *k1)
            .and(&buf.x)
            .apply(|k1, &x_| { *k1 = k1.mul_real(dt) + x_; });
        let k2 = self.f.rhs(k1);
        Zip::from(&mut *k2)
            .and(&buf.x)
            .and(&buf.k1)
            .apply(|k2, &x_, &k1_| { *k2 = x_ + (k1_ + *k2).mul_real(dt_2); });
        k2
    }
}

pub struct RK4Buffer<A, D> {
    x: Array<A, D>,
    k1: Array<A, D>,
    k2: Array<A, D>,
    k3: Array<A, D>,
}

impl<A, D, F> WithBuffer for RK4<A, D, F>
    where A: Scalar,
          D: Dimension,
          F: Explicit<D, Scalar = A> + TimeStep<Time = A::Real> + ModelSize<D>
{
    type Buffer = RK4Buffer<A, D>;

    fn new_buffer(&self) -> Self::Buffer {
        RK4Buffer {
            x: Array::zeros(self.f.model_size()),
            k1: Array::zeros(self.f.model_size()),
            k2: Array::zeros(self.f.model_size()),
            k3: Array::zeros(self.f.model_size()),
        }
    }
}

impl<A, D, F> TimeEvolution<D> for RK4<A, D, F>
    where A: Scalar,
          D: Dimension,
          F: Explicit<D, Scalar = A> + TimeStep<Time = A::Real> + ModelSize<D>
{
    type Scalar = A;

    fn iterate<'a, S>(&self,
                      mut x: &'a mut ArrayBase<S, D>,
                      mut buf: &mut RK4Buffer<A, D>)
                      -> &'a mut ArrayBase<S, D>
        where S: DataMut<Elem = A>
    {
        let two = into_scalar(2.0);
        let dt = self.dt;
        let dt_2 = self.dt * into_scalar(0.5);
        let dt_6 = self.dt / into_scalar(6.0);
        buf.x.zip_mut_with(x, |buf, x| *buf = *x);
        // k1
        let mut k1 = self.f.rhs(x);
        buf.k1.zip_mut_with(k1, |buf, k1| *buf = *k1);
        Zip::from(&mut *k1)
            .and(&buf.x)
            .apply(|k1, &x| { *k1 = k1.mul_real(dt_2) + x; });
        // k2
        let mut k2 = self.f.rhs(k1);
        buf.k2.zip_mut_with(k2, |buf, k| *buf = *k);
        Zip::from(&mut *k2)
            .and(&buf.x)
            .apply(|k2, &x| { *k2 = x + k2.mul_real(dt_2); });
        // k3
        let mut k3 = self.f.rhs(k2);
        buf.k3.zip_mut_with(k3, |buf, k| *buf = *k);
        Zip::from(&mut *k3)
            .and(&buf.x)
            .apply(|k3, &x| { *k3 = x + k3.mul_real(dt); });
        let mut k4 = self.f.rhs(k3);
        Zip::from(&mut *k4)
            .and(&buf.x)
            .and(&buf.k1)
            .and(&buf.k2)
            .and(&buf.k3)
            .apply(|k4, &x, &k1, &k2, &k3| {
                       *k4 = x + (k1 + (k2 + k3).mul_real(two) + *k4).mul_real(dt_6);
                   });
        k4
    }
}
