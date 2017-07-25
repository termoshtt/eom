//! Define explicit schemes

use ndarray::*;
use ndarray_linalg::*;
use super::traits::*;

macro_rules! def_explicit {
    ($method:ident, $constructor:ident) => {

#[derive(new)]
pub struct $method<F, Time: RealScalar> {
    f: F,
    dt: Time,
}

impl<F, D, Time> ModelSize<D> for $method<F, Time>
    where F: ModelSize<D>,
          D: Dimension,
          Time: RealScalar
{
    fn model_size(&self) -> D::Pattern {
        self.f.model_size()
    }
}

impl<F, Time: RealScalar> TimeStep for $method<F, Time> {
    type Time = Time;
    fn get_dt(&self) -> Self::Time {
        self.dt
    }
    fn set_dt(&mut self, dt: Self::Time) {
        self.dt = dt;
    }
}

pub fn $constructor<F, Time: RealScalar>(f: F, dt: Time) -> $method<F, Time> {
    $method::new(f, dt)
}

impl<A, D, F> TimeEvolution<A, D> for $method<F, A::Real>
    where A: Scalar,
          D: Dimension,
          F: Explicit<OwnedRepr<A>, D, Scalar=A, Time=A::Real>
           + Explicit<OwnedRcRepr<A>, D, Scalar=A, Time=A::Real>
           + for<'a> Explicit<ViewRepr<&'a mut A>, D, Scalar=A, Time=A::Real>
{}

}} // def_explicit

def_explicit!(Euler, euler);
def_explicit!(Heun, heun);
def_explicit!(RK4, rk4);

impl<A, S, D, F> TimeEvolutionBase<S, D> for Euler<F, F::Time>
    where A: Scalar,
          S: DataMut<Elem = A>,
          D: Dimension,
          F: Explicit<S, D, Time = A::Real, Scalar = A>
{
    type Scalar = F::Scalar;

    fn iterate<'a>(&self, mut x: &'a mut ArrayBase<S, D>) -> &'a mut ArrayBase<S, D> {
        let x_ = x.to_owned();
        let fx = self.f.rhs(x);
        Zip::from(&mut *fx)
            .and(&x_)
            .apply(|vfx, vx| { *vfx = *vx + vfx.mul_real(self.dt); });
        fx
    }
}

impl<A, S, D, F> TimeEvolutionBase<S, D> for Heun<F, F::Time>
    where A: Scalar,
          S: DataMut<Elem = A>,
          D: Dimension,
          F: Explicit<S, D, Time = A::Real, Scalar = A>
{
    type Scalar = F::Scalar;

    fn iterate<'a>(&self, mut x: &'a mut ArrayBase<S, D>) -> &'a mut ArrayBase<S, D> {
        let dt = self.dt;
        let dt_2 = self.dt * into_scalar(0.5);
        // calc
        let x_ = x.to_owned();
        let k1 = self.f.rhs(x);
        let k1_ = k1.to_owned();
        Zip::from(&mut *k1)
            .and(&x_)
            .apply(|k1, &x_| { *k1 = k1.mul_real(dt) + x_; });
        let k2 = self.f.rhs(k1);
        Zip::from(&mut *k2)
            .and(&x_)
            .and(&k1_)
            .apply(|k2, &x_, &k1_| { *k2 = x_ + (k1_ + *k2).mul_real(dt_2); });
        k2
    }
}

impl<A, S, D, F> TimeEvolutionBase<S, D> for RK4<F, F::Time>
    where A: Scalar,
          S: DataMut<Elem = A>,
          D: Dimension,
          F: Explicit<S, D, Time = A::Real, Scalar = A>
{
    type Scalar = F::Scalar;

    fn iterate<'a>(&self, mut x: &'a mut ArrayBase<S, D>) -> &'a mut ArrayBase<S, D> {
        let dt = self.dt;
        let dt_2 = self.dt * into_scalar(0.5);
        let dt_6 = self.dt / into_scalar(6.0);
        let x_ = x.to_owned();
        // k1
        let mut k1 = self.f.rhs(x);
        let k1_ = k1.to_owned();
        Zip::from(&mut *k1)
            .and(&x_)
            .apply(|k1, &x_| { *k1 = k1.mul_real(dt_2) + x_; });
        // k2
        let mut k2 = self.f.rhs(k1);
        let k2_ = k2.to_owned();
        Zip::from(&mut *k2)
            .and(&x_)
            .apply(|k2, &x_| { *k2 = x_ + k2.mul_real(dt_2); });
        // k3
        let mut k3 = self.f.rhs(k2);
        let k3_ = k3.to_owned();
        Zip::from(&mut *k3)
            .and(&x_)
            .apply(|k3, &x_| { *k3 = x_ + k3.mul_real(dt); });
        let mut k4 = self.f.rhs(k3);
        Zip::from(&mut *k4)
            .and(&x_)
            .and(&k1_)
            .and(&k2_)
            .and(&k3_)
            .apply(|k4, &x_, &k1_, &k2_, &k3_| {
                       *k4 = x_ +
                             (k1_ + (k2_ + k3_).mul_real(into_scalar(2.0)) + *k4).mul_real(dt_6);
                   });
        k4
    }
}

pub struct RK4Buffer<A, D> {
    x: Array<A, D>,
    k1: Array<A, D>,
    k2: Array<A, D>,
    k3: Array<A, D>,
}

impl<A, S, D, F> TimeEvolutionBuffered<S, D> for RK4<F, F::Time>
    where A: Scalar,
          S: DataMut<Elem = A>,
          D: Dimension,
          F: Explicit<S, D, Time = A::Real, Scalar = A>
{
    type Scalar = F::Scalar;
    type Buffer = RK4Buffer<A, D>;

    fn get_buffer(&self) -> Self::Buffer {
        RK4Buffer {
            x: Array::zeros(self.model_size()),
            k1: Array::zeros(self.model_size()),
            k2: Array::zeros(self.model_size()),
            k3: Array::zeros(self.model_size()),
        }
    }

    fn iterate_buf<'a>(&self,
                       mut x: &'a mut ArrayBase<S, D>,
                       mut buf: &mut Self::Buffer)
                       -> &'a mut ArrayBase<S, D> {
        let dt = self.dt;
        let dt_2 = self.dt * into_scalar(0.5);
        let dt_6 = self.dt / into_scalar(6.0);
        buf.x.assign(x);
        // k1
        let mut k1 = self.f.rhs(x);
        buf.k1.assign(k1);
        Zip::from(&mut *k1)
            .and(&buf.x)
            .apply(|k1, &x| { *k1 = k1.mul_real(dt_2) + x; });
        // k2
        let mut k2 = self.f.rhs(k1);
        buf.k2.assign(k2);
        Zip::from(&mut *k2)
            .and(&buf.x)
            .apply(|k2, &x| { *k2 = x + k2.mul_real(dt_2); });
        // k3
        let mut k3 = self.f.rhs(k2);
        buf.k3.assign(k3);
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
                       *k4 = x + (k1 + (k2 + k3).mul_real(into_scalar(2.0)) + *k4).mul_real(dt_6);
                   });
        k4
    }
}
