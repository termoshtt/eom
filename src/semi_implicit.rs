//! semi-implicit schemes

use ndarray::*;
use ndarray_linalg::*;

use super::traits::*;

/// Linear ODE with diagonalized matrix (exactly solvable)
#[derive(Debug, Clone)]
pub struct Diagonal<F: SemiImplicit> {
    exp_diag: Array<F::Scalar, F::Dim>,
    diag: Array<F::Scalar, F::Dim>,
    dt: <F::Scalar as AssociatedReal>::Real,
}

impl<F: SemiImplicit> TimeStep for Diagonal<F> {
    type Time = <F::Scalar as AssociatedReal>::Real;

    fn get_dt(&self) -> Self::Time {
        self.dt
    }
    fn set_dt(&mut self, dt: Self::Time) {
        Zip::from(&mut self.exp_diag)
            .and(&self.diag)
            .apply(|a, &b| {
                *a = b.mul_real(dt).exp();
            });
    }
}

impl<F: SemiImplicit> ModelSpec for Diagonal<F> {
    type Scalar = F::Scalar;
    type Dim = F::Dim;

    fn model_size(&self) -> <Self::Dim as Dimension>::Pattern {
        self.exp_diag.dim()
    }
}

impl<F: SemiImplicit> TimeEvolution for Diagonal<F> {
    fn iterate<'a, S>(
        &mut self,
        x: &'a mut ArrayBase<S, Self::Dim>,
    ) -> &'a mut ArrayBase<S, Self::Dim>
    where
        S: DataMut<Elem = Self::Scalar>,
    {
        for (val, d) in x.iter_mut().zip(self.exp_diag.iter()) {
            *val = *val * *d;
        }
        x
    }
}

impl<F: SemiImplicit> Diagonal<F> {
    fn new(f: F, dt: <Self as TimeStep>::Time) -> Self {
        let diag = f.diag();
        let mut exp_diag = diag.to_owned();
        for v in exp_diag.iter_mut() {
            *v = v.mul_real(dt).exp();
        }
        Diagonal {
            exp_diag: exp_diag,
            diag: diag,
            dt: dt,
        }
    }
}

#[derive(Debug, Clone)]
pub struct DiagRK4<F: SemiImplicit> {
    nlin: F,
    lin: Diagonal<F>,
    dt: <Diagonal<F> as TimeStep>::Time,
    x: Array<F::Scalar, F::Dim>,
    lx: Array<F::Scalar, F::Dim>,
    k1: Array<F::Scalar, F::Dim>,
    k2: Array<F::Scalar, F::Dim>,
    k3: Array<F::Scalar, F::Dim>,
}

impl<F: SemiImplicit> Scheme for DiagRK4<F> {
    type Core = F;
    fn new(nlin: F, dt: Self::Time) -> Self {
        let lin = Diagonal::new(nlin.clone(), dt / into_scalar(2.0));
        let x = Array::zeros(lin.model_size());
        let lx = Array::zeros(lin.model_size());
        let k1 = Array::zeros(lin.model_size());
        let k2 = Array::zeros(lin.model_size());
        let k3 = Array::zeros(lin.model_size());
        DiagRK4 {
            nlin,
            lin,
            dt,
            x,
            lx,
            k1,
            k2,
            k3,
        }
    }
    fn core(&self) -> &Self::Core {
        &self.nlin
    }
    fn core_mut(&mut self) -> &mut Self::Core {
        &mut self.nlin
    }
}

impl<F: SemiImplicit> TimeStep for DiagRK4<F> {
    type Time = <Diagonal<F> as TimeStep>::Time;

    fn get_dt(&self) -> Self::Time {
        self.dt
    }

    fn set_dt(&mut self, dt: Self::Time) {
        self.lin.set_dt(dt / into_scalar(2.0));
    }
}

impl<F: SemiImplicit> ModelSpec for DiagRK4<F> {
    type Scalar = F::Scalar;
    type Dim = F::Dim;

    fn model_size(&self) -> <Self::Dim as Dimension>::Pattern {
        self.nlin.model_size() // TODO check
    }
}

impl<F: SemiImplicit> TimeEvolution for DiagRK4<F> {
    fn iterate<'a, S>(
        &mut self,
        x: &'a mut ArrayBase<S, Self::Dim>,
    ) -> &'a mut ArrayBase<S, Self::Dim>
    where
        S: DataMut<Elem = Self::Scalar>,
    {
        // constants
        let dt = self.dt;
        let dt_2 = self.dt / into_scalar(2.0);
        let dt_3 = self.dt / into_scalar(3.0);
        let dt_6 = self.dt / into_scalar(6.0);
        // operators
        let l = &mut self.lin;
        let f = &mut self.nlin;
        // calc
        self.x.zip_mut_with(x, |buf, x| *buf = *x);
        self.lx.zip_mut_with(x, |buf, lx| *buf = *lx);
        l.iterate(&mut self.lx);
        let k1 = f.nlin(x);
        self.k1.zip_mut_with(k1, |buf, k1| *buf = *k1);
        Zip::from(&mut *k1).and(&self.x).apply(|k1, &x_| {
            *k1 = x_ + k1.mul_real(dt_2);
        });
        let k2 = f.nlin(l.iterate(k1));
        self.k2.zip_mut_with(k2, |buf, k| *buf = *k);
        Zip::from(&mut *k2).and(&self.lx).apply(|k2, &lx| {
            *k2 = lx + k2.mul_real(dt_2);
        });
        let k3 = f.nlin(k2);
        self.k3.zip_mut_with(k3, |buf, k| *buf = *k);
        Zip::from(&mut *k3).and(&self.lx).apply(|k3, &lx| {
            *k3 = lx + k3.mul_real(dt);
        });
        let k4 = f.nlin(l.iterate(k3));
        Zip::from(&mut self.x)
            .and(&self.k1)
            .apply(|x_, k1_| *x_ = *x_ + k1_.mul_real(dt_6));
        l.iterate(&mut self.x);
        Zip::from(&mut self.x)
            .and(&self.k2)
            .and(&self.k3)
            .apply(|x_, &k2_, &k3_| *x_ = *x_ + (k2_ + k3_).mul_real(dt_3));
        l.iterate(&mut self.x);
        Zip::from(&mut *k4).and(&self.x).apply(|k4, &x_| {
            *k4 = x_ + k4.mul_real(dt_6);
        });
        k4
    }
}
