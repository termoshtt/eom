use ndarray::*;
use ndarray_linalg::*;
use super::traits::*;

pub struct TimeSeries<'a, TEO, S>
where
    S: DataMut,
    TEO: TimeEvolution + 'a,
{
    state: ArrayBase<S, TEO::Dim>,
    teo: &'a mut TEO,
}

pub fn time_series<'a, TEO, S>(
    x0: ArrayBase<S, TEO::Dim>,
    teo: &'a mut TEO,
) -> TimeSeries<'a, TEO, S>
where
    S: DataMut,
    TEO: TimeEvolution,
{
    TimeSeries {
        state: x0,
        teo: teo,
    }
}

impl<'a, TEO, A, S> TimeSeries<'a, TEO, S>
where
    A: Scalar,
    S: DataMut<Elem = A> + DataClone,
    TEO: TimeEvolution<Scalar = A>,
{
    pub fn iterate(&mut self) {
        self.teo.iterate(&mut self.state);
    }
}

impl<'a, TEO, A, S> Iterator for TimeSeries<'a, TEO, S>
where
    A: Scalar,
    S: DataMut<Elem = A> + DataClone,
    TEO: TimeEvolution<Scalar = A>,
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
/// let teo = explicit::RK4::new(ode::Lorenz63::default(), 0.01);
/// let nstep = nstep(teo, 10);
/// ```
#[derive(Debug, Clone)]
pub struct NStep<TEO> {
    teo: TEO,
    n: usize,
}

pub fn nstep<TEO>(teo: TEO, n: usize) -> NStep<TEO> {
    NStep { teo, n }
}

impl<TEO> ModelSpec for NStep<TEO>
where
    TEO: ModelSpec,
{
    type Scalar = TEO::Scalar;
    type Dim = TEO::Dim;

    fn model_size(&self) -> <Self::Dim as Dimension>::Pattern {
        self.teo.model_size()
    }
}

impl<TEO> TimeStep for NStep<TEO>
where
    TEO: TimeStep,
{
    type Time = TEO::Time;

    fn get_dt(&self) -> Self::Time {
        self.teo.get_dt() * into_scalar(self.n as f64)
    }

    fn set_dt(&mut self, dt: Self::Time) {
        self.teo.set_dt(dt / into_scalar(self.n as f64));
    }
}

impl<TEO> TimeEvolution for NStep<TEO>
where
    TEO: TimeEvolution,
{
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
