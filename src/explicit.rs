//! Define explicit schemes

use ndarray::*;
use ndarray_linalg::*;
use super::traits::*;

#[derive(Debug, Clone)]
pub struct Euler<F: Explicit> {
    f: F,
    dt: <F::Scalar as AssociatedReal>::Real,
    x: Array<F::Scalar, F::Dim>,
}

impl<A: Scalar, F: Explicit<Scalar = A>> TimeStep for Euler<F> {
    type Time = A::Real;

    fn get_dt(&self) -> Self::Time {
        self.dt
    }

    fn set_dt(&mut self, dt: Self::Time) {
        self.dt = dt;
    }
}

impl<A: Scalar, F: Explicit<Scalar = A>> Euler<F> {
    pub fn new(f: F, dt: A::Real) -> Self {
        let x = Array::zeros(f.model_size());
        Self { f, dt, x }
    }
}

impl<F: Explicit> ModelSpec for Euler<F> {
    type Scalar = F::Scalar;
    type Dim = F::Dim;
    fn model_size(&self) -> <Self::Dim as Dimension>::Pattern {
        self.f.model_size()
    }
}

impl<F: Explicit + ModelSpec> TimeEvolution for Euler<F> {
    fn iterate<'a, S>(&mut self, x: &'a mut ArrayBase<S, F::Dim>) -> &'a mut ArrayBase<S, Self::Dim>
    where
        S: DataMut<Elem = Self::Scalar>,
    {
        self.x.zip_mut_with(x, |buf, x| *buf = *x);
        let fx = self.f.rhs(x);
        Zip::from(&mut *fx).and(&self.x).apply(|vfx, vx| {
            *vfx = *vx + vfx.mul_real(self.dt);
        });
        fx
    }
}

#[derive(Debug, Clone)]
pub struct Heun<F: Explicit> {
    f: F,
    dt: <F::Scalar as AssociatedReal>::Real,
    x: Array<F::Scalar, F::Dim>,
    k1: Array<F::Scalar, F::Dim>,
}

impl<A: Scalar, F: Explicit<Scalar = A>> TimeStep for Heun<F> {
    type Time = A::Real;

    fn get_dt(&self) -> Self::Time {
        self.dt
    }

    fn set_dt(&mut self, dt: Self::Time) {
        self.dt = dt;
    }
}

impl<A: Scalar, F: Explicit<Scalar = A>> Heun<F> {
    pub fn new(f: F, dt: A::Real) -> Self {
        let x = Array::zeros(f.model_size());
        let k1 = Array::zeros(f.model_size());
        Self { f, dt, x, k1 }
    }
}

impl<F: Explicit> ModelSpec for Heun<F> {
    type Scalar = F::Scalar;
    type Dim = F::Dim;
    fn model_size(&self) -> <Self::Dim as Dimension>::Pattern {
        self.f.model_size()
    }
}

impl<F: Explicit + ModelSpec> TimeEvolution for Heun<F> {
    fn iterate<'a, S>(&mut self, x: &'a mut ArrayBase<S, F::Dim>) -> &'a mut ArrayBase<S, Self::Dim>
    where
        S: DataMut<Elem = Self::Scalar>,
    {
        let dt = self.dt;
        let dt_2 = self.dt * into_scalar(0.5);
        // calc
        self.x.zip_mut_with(x, |buf, x| *buf = *x);
        let k1 = self.f.rhs(x);
        self.k1.zip_mut_with(k1, |buf, k1| *buf = *k1);
        Zip::from(&mut *k1).and(&self.x).apply(|k1, &x_| {
            *k1 = k1.mul_real(dt) + x_;
        });
        let k2 = self.f.rhs(k1);
        Zip::from(&mut *k2)
            .and(&self.x)
            .and(&self.k1)
            .apply(|k2, &x_, &k1_| {
                *k2 = x_ + (k1_ + *k2).mul_real(dt_2);
            });
        k2
    }
}

#[derive(Debug, Clone)]
pub struct RK4<F: Explicit> {
    f: F,
    dt: <F::Scalar as AssociatedReal>::Real,
    x: Array<F::Scalar, F::Dim>,
    k1: Array<F::Scalar, F::Dim>,
    k2: Array<F::Scalar, F::Dim>,
    k3: Array<F::Scalar, F::Dim>,
}

impl<A: Scalar, F: Explicit<Scalar = A>> TimeStep for RK4<F> {
    type Time = A::Real;

    fn get_dt(&self) -> Self::Time {
        self.dt
    }

    fn set_dt(&mut self, dt: Self::Time) {
        self.dt = dt;
    }
}

impl<A: Scalar, F: Explicit<Scalar = A>> RK4<F> {
    pub fn new(f: F, dt: A::Real) -> Self {
        let x = Array::zeros(f.model_size());
        let k1 = Array::zeros(f.model_size());
        let k2 = Array::zeros(f.model_size());
        let k3 = Array::zeros(f.model_size());
        Self {
            f,
            dt,
            x,
            k1,
            k2,
            k3,
        }
    }
}

impl<F: Explicit> ModelSpec for RK4<F> {
    type Scalar = F::Scalar;
    type Dim = F::Dim;
    fn model_size(&self) -> <Self::Dim as Dimension>::Pattern {
        self.f.model_size()
    }
}

impl<F: Explicit + ModelSpec> TimeEvolution for RK4<F> {
    fn iterate<'a, S>(&mut self, x: &'a mut ArrayBase<S, F::Dim>) -> &'a mut ArrayBase<S, Self::Dim>
    where
        S: DataMut<Elem = Self::Scalar>,
    {
        let two = into_scalar(2.0);
        let dt = self.dt;
        let dt_2 = self.dt * into_scalar(0.5);
        let dt_6 = self.dt / into_scalar(6.0);
        self.x.zip_mut_with(x, |buf, x| *buf = *x);
        // k1
        let k1 = self.f.rhs(x);
        self.k1.zip_mut_with(k1, |buf, k1| *buf = *k1);
        Zip::from(&mut *k1).and(&self.x).apply(|k1, &x| {
            *k1 = k1.mul_real(dt_2) + x;
        });
        // k2
        let k2 = self.f.rhs(k1);
        self.k2.zip_mut_with(k2, |buf, k| *buf = *k);
        Zip::from(&mut *k2).and(&self.x).apply(|k2, &x| {
            *k2 = x + k2.mul_real(dt_2);
        });
        // k3
        let k3 = self.f.rhs(k2);
        self.k3.zip_mut_with(k3, |buf, k| *buf = *k);
        Zip::from(&mut *k3).and(&self.x).apply(|k3, &x| {
            *k3 = x + k3.mul_real(dt);
        });
        let k4 = self.f.rhs(k3);
        Zip::from(&mut *k4)
            .and(&self.x)
            .and(&self.k1)
            .and(&self.k2)
            .and(&self.k3)
            .apply(|k4, &x, &k1, &k2, &k3| {
                *k4 = x + (k1 + (k2 + k3).mul_real(two) + *k4).mul_real(dt_6);
            });
        k4
    }
}
