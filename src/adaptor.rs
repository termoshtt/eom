use ndarray::*;
use ndarray_linalg::*;
use super::traits::*;

pub fn accuracy<A, D, F, Sc>(
    mut teo: Sc,
    init: Array<A, D>,
    dt_base: A::Real,
    step_base: usize,
    num_scale: u32,
) -> Vec<(Sc::Time, A::Real)>
where
    A: Scalar,
    D: Dimension,
    Sc: Scheme<F, Scalar = A, Dim = D, Time = A::Real>,
{
    let data: Vec<_> = (0..num_scale)
        .map(|n| {
            let rate = 2_usize.pow(n);
            let dt = dt_base / into_scalar(rate as f64);
            let t = step_base * rate;
            teo.set_dt(dt);
            (dt, iterate(&mut teo, init.clone(), t))
        })
        .collect();
    data.windows(2)
        .map(|w| {
            let dt = w[0].0;
            let dev = (&w[1].1 - &w[0].1).norm();
            (dt, dev)
        })
        .collect()
}

pub fn iterate<S, TEO>(
    teo: &mut TEO,
    x0: ArrayBase<S, TEO::Dim>,
    step: usize,
) -> ArrayBase<S, TEO::Dim>
where
    S: DataMut<Elem = TEO::Scalar> + DataClone,
    TEO: TimeEvolution,
{
    let mut ts = time_series(x0, teo);
    ts.nth(step - 1).unwrap()
}

pub struct TimeSeries<'a, S, TEO>
where
    S: DataMut<Elem = TEO::Scalar> + DataClone,
    TEO: TimeEvolution + 'a,
{
    state: ArrayBase<S, TEO::Dim>,
    teo: &'a mut TEO,
}

pub fn time_series<'a, S, TEO>(
    x0: ArrayBase<S, TEO::Dim>,
    teo: &'a mut TEO,
) -> TimeSeries<'a, S, TEO>
where
    S: DataMut<Elem = TEO::Scalar> + DataClone,
    TEO: TimeEvolution,
{
    TimeSeries {
        state: x0,
        teo: teo,
    }
}

impl<'a, S, TEO> TimeSeries<'a, S, TEO>
where
    S: DataMut<Elem = TEO::Scalar> + DataClone,
    TEO: TimeEvolution,
{
    pub fn iterate(&mut self) {
        self.teo.iterate(&mut self.state);
    }
}

impl<'a, S, TEO> Iterator for TimeSeries<'a, S, TEO>
where
    S: DataMut<Elem = TEO::Scalar> + DataClone,
    TEO: TimeEvolution,
{
    type Item = ArrayBase<S, TEO::Dim>;
    fn next(&mut self) -> Option<Self::Item> {
        self.iterate();
        Some(self.state.clone())
    }
}

/// N-step adaptor
///
/// ```rust
/// use eom::*;
/// use eom::traits::*;
/// let teo = explicit::RK4::new(ode::Lorenz63::default(), 0.01);
/// let nstep = adaptor::nstep(teo, 10);
/// ```
#[derive(Debug, Clone)]
pub struct NStep<TEO: TimeEvolution> {
    teo: TEO,
    n: usize,
}

pub fn nstep<TEO: TimeEvolution>(teo: TEO, n: usize) -> NStep<TEO> {
    NStep { teo, n }
}

impl<TEO: TimeEvolution> ModelSpec for NStep<TEO> {
    type Scalar = TEO::Scalar;
    type Dim = TEO::Dim;

    fn model_size(&self) -> <Self::Dim as Dimension>::Pattern {
        self.teo.model_size()
    }
}

impl<TEO: TimeEvolution> TimeStep for NStep<TEO> {
    type Time = TEO::Time;

    fn get_dt(&self) -> Self::Time {
        self.teo.get_dt() * into_scalar(self.n as f64)
    }

    fn set_dt(&mut self, dt: Self::Time) {
        self.teo.set_dt(dt / into_scalar(self.n as f64));
    }
}

impl<TEO: TimeEvolution> TimeEvolution for NStep<TEO> {
    fn iterate<'a, S>(
        &mut self,
        x: &'a mut ArrayBase<S, TEO::Dim>,
    ) -> &'a mut ArrayBase<S, TEO::Dim>
    where
        S: DataMut<Elem = TEO::Scalar>,
    {
        for _ in 0..self.n {
            self.teo.iterate(x);
        }
        x
    }
}
