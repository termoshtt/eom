
use ndarray::*;
use ndarray_linalg::*;
use super::traits::*;

pub struct TimeSeries<'a, TEO, S, D>
    where S: DataMut,
          D: Dimension,
          TEO: TimeEvolutionBase<S, D> + 'a
{
    state: ArrayBase<S, D>,
    teo: &'a TEO,
}

pub fn time_series<'a, TEO, S, D>(x0: ArrayBase<S, D>, teo: &'a TEO) -> TimeSeries<'a, TEO, S, D>
    where S: DataMut,
          D: Dimension,
          TEO: TimeEvolutionBase<S, D>
{
    TimeSeries {
        state: x0,
        teo: teo,
    }
}

impl<'a, TEO, S, D> TimeSeries<'a, TEO, S, D>
    where S: DataMut + DataClone,
          D: Dimension,
          TEO: TimeEvolutionBase<S, D>
{
    pub fn iterate(&mut self) {
        self.teo.iterate(&mut self.state);
    }
}

impl<'a, TEO, S, D> Iterator for TimeSeries<'a, TEO, S, D>
    where S: DataMut + DataClone,
          D: Dimension,
          TEO: TimeEvolutionBase<S, D>
{
    type Item = ArrayBase<S, D>;
    fn next(&mut self) -> Option<Self::Item> {
        self.iterate();
        Some(self.state.clone())
    }
}


/// N-step adaptor
///
/// ```rust
/// use ndarray_odeint::*;
/// let teo = explicit::rk4(model::Lorenz63::default(), 0.01);
/// let nstep = nstep(teo, 10);
/// ```
pub struct NStep<TEO> {
    teo: TEO,
    n: usize,
}

pub fn nstep<TEO>(teo: TEO, n: usize) -> NStep<TEO> {
    NStep { teo, n }
}

impl<TEO, D> ModelSize<D> for NStep<TEO>
    where TEO: ModelSize<D>,
          D: Dimension
{
    fn model_size(&self) -> D::Pattern {
        self.teo.model_size()
    }
}

impl<TEO> TimeStep for NStep<TEO>
    where TEO: TimeStep
{
    type Time = TEO::Time;

    fn get_dt(&self) -> Self::Time {
        self.teo.get_dt() * into_scalar(self.n as f64)
    }

    fn set_dt(&mut self, dt: Self::Time) {
        self.teo.set_dt(dt / into_scalar(self.n as f64));
    }
}

impl<TEO, S, D> TimeEvolutionBase<S, D> for NStep<TEO>
    where TEO: TimeEvolutionBase<S, D>,
          S: DataMut,
          D: Dimension
{
    type Scalar = TEO::Scalar;
    type Time = TEO::Time;

    fn iterate<'a>(&self, x: &'a mut ArrayBase<S, D>) -> &'a mut ArrayBase<S, D> {
        for _ in 0..self.n {
            self.teo.iterate(x);
        }
        x
    }
}

impl<TEO, A, D> TimeEvolution<A, D> for NStep<TEO>
    where A: Scalar,
          D: Dimension,
          TEO: TimeEvolution<A, D>
{
}
