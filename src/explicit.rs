//! Define explicit schemes

use ndarray::*;
use ndarray_linalg::*;
use super::traits::*;

macro_rules! def_explicit {
    ($method:ident, $constructor:ident) => {

pub struct $method<F: ExplicitBuf> {
    f: F,
    dt: <F::Scalar as AssociatedReal>::Real,
}

impl<D: Dimension, F: ExplicitBuf<Dim = D>> ModelSpec for $method<F> {
    type Scalar = F::Scalar;
    type Dim = D;

    fn model_size(&self) -> D::Pattern {
        self.f.model_size()
    }
}

impl<A: Scalar, F: ExplicitBuf<Scalar = A>> TimeStep for $method<F> {
    type Time = A::Real;
    fn get_dt(&self) -> Self::Time {
        self.dt
    }
    fn set_dt(&mut self, dt: Self::Time) {
        self.dt = dt;
    }
}

pub fn $constructor<A, F>(f: F, dt: A::Real) -> $method<F>
where A: Scalar,
      F: ExplicitBuf<Scalar = A>
{
    $method { f: f, dt: dt }
}

}} // def_explicit

def_explicit!(Euler, euler);
def_explicit!(Heun, heun);
def_explicit!(RK4, rk4);

pub struct EulerBuffer<ExBuf, Arr> {
    ex: ExBuf,
    x: Arr,
}

impl<F> BufferSpec for Euler<F>
    where F: ExplicitBuf + ModelSpec
{
    type Buffer = EulerBuffer<F::Buffer, Array<F::Scalar, F::Dim>>;

    fn new_buffer(&self) -> Self::Buffer {
        EulerBuffer {
            ex: self.f.new_buffer(),
            x: Array::zeros(self.f.model_size()),
        }
    }
}

impl<F> TimeEvolution for Euler<F>
    where F: ExplicitBuf + ModelSpec
{
    fn iterate<'a, S>(&self,
                      x: &'a mut ArrayBase<S, F::Dim>,
                      buf: &mut Self::Buffer)
                      -> &'a mut ArrayBase<S, F::Dim>
        where S: DataMut<Elem = Self::Scalar>
    {
        buf.x.zip_mut_with(x, |buf, x| *buf = *x);
        let fx = self.f.rhs(x, &mut buf.ex);
        Zip::from(&mut *fx)
            .and(&buf.x)
            .apply(|vfx, vx| { *vfx = *vx + vfx.mul_real(self.dt); });
        fx
    }
}

pub struct HeunBuffer<ExBuf, Arr> {
    ex: ExBuf,
    x: Arr,
    k1: Arr,
}

impl<F: ExplicitBuf + ModelSpec> BufferSpec for Heun<F> {
    type Buffer = HeunBuffer<F::Buffer, Array<F::Scalar, F::Dim>>;

    fn new_buffer(&self) -> Self::Buffer {
        HeunBuffer {
            ex: self.f.new_buffer(),
            x: Array::zeros(self.f.model_size()),
            k1: Array::zeros(self.f.model_size()),
        }
    }
}

impl<F: ExplicitBuf + ModelSpec> TimeEvolution for Heun<F> {
    fn iterate<'a, S>(&self,
                      x: &'a mut ArrayBase<S, F::Dim>,
                      buf: &mut Self::Buffer)
                      -> &'a mut ArrayBase<S, F::Dim>
        where S: DataMut<Elem = Self::Scalar>
    {
        let dt = self.dt;
        let dt_2 = self.dt * into_scalar(0.5);
        // calc
        buf.x.zip_mut_with(x, |buf, x| *buf = *x);
        let k1 = self.f.rhs(x, &mut buf.ex);
        buf.k1.zip_mut_with(k1, |buf, k1| *buf = *k1);
        Zip::from(&mut *k1)
            .and(&buf.x)
            .apply(|k1, &x_| { *k1 = k1.mul_real(dt) + x_; });
        let k2 = self.f.rhs(k1, &mut buf.ex);
        Zip::from(&mut *k2)
            .and(&buf.x)
            .and(&buf.k1)
            .apply(|k2, &x_, &k1_| { *k2 = x_ + (k1_ + *k2).mul_real(dt_2); });
        k2
    }
}

pub struct RK4Buffer<ExBuf, Arr> {
    ex: ExBuf,
    x: Arr,
    k1: Arr,
    k2: Arr,
    k3: Arr,
}

impl<F: ExplicitBuf + ModelSpec> BufferSpec for RK4<F> {
    type Buffer = RK4Buffer<F::Buffer, Array<F::Scalar, F::Dim>>;

    fn new_buffer(&self) -> Self::Buffer {
        RK4Buffer {
            ex: self.f.new_buffer(),
            x: Array::zeros(self.f.model_size()),
            k1: Array::zeros(self.f.model_size()),
            k2: Array::zeros(self.f.model_size()),
            k3: Array::zeros(self.f.model_size()),
        }
    }
}

impl<F: ExplicitBuf + ModelSpec> TimeEvolution for RK4<F> {
    fn iterate<'a, S>(&self,
                      x: &'a mut ArrayBase<S, F::Dim>,
                      buf: &mut Self::Buffer)
                      -> &'a mut ArrayBase<S, F::Dim>
        where S: DataMut<Elem = Self::Scalar>
    {
        let two = into_scalar(2.0);
        let dt = self.dt;
        let dt_2 = self.dt * into_scalar(0.5);
        let dt_6 = self.dt / into_scalar(6.0);
        buf.x.zip_mut_with(x, |buf, x| *buf = *x);
        // k1
        let k1 = self.f.rhs(x, &mut buf.ex);
        buf.k1.zip_mut_with(k1, |buf, k1| *buf = *k1);
        Zip::from(&mut *k1)
            .and(&buf.x)
            .apply(|k1, &x| { *k1 = k1.mul_real(dt_2) + x; });
        // k2
        let k2 = self.f.rhs(k1, &mut buf.ex);
        buf.k2.zip_mut_with(k2, |buf, k| *buf = *k);
        Zip::from(&mut *k2)
            .and(&buf.x)
            .apply(|k2, &x| { *k2 = x + k2.mul_real(dt_2); });
        // k3
        let k3 = self.f.rhs(k2, &mut buf.ex);
        buf.k3.zip_mut_with(k3, |buf, k| *buf = *k);
        Zip::from(&mut *k3)
            .and(&buf.x)
            .apply(|k3, &x| { *k3 = x + k3.mul_real(dt); });
        let k4 = self.f.rhs(k3, &mut buf.ex);
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
