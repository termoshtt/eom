
use ndarray::*;
use ndarray_linalg::*;
use super::traits::*;

pub struct TimeSeries<'a, TEO, S, D>
    where S: DataMut,
          D: Dimension,
          TEO: TimeEvolution<D> + 'a
{
    state: ArrayBase<S, D>,
    buf: TEO::Buffer,
    teo: &'a TEO,
}

pub fn time_series<'a, TEO, S, D>(x0: ArrayBase<S, D>, teo: &'a TEO) -> TimeSeries<'a, TEO, S, D>
    where S: DataMut,
          D: Dimension,
          TEO: TimeEvolution<D>
{
    TimeSeries {
        state: x0,
        buf: teo.new_buffer(),
        teo: teo,
    }
}

impl<'a, TEO, A, S, D> TimeSeries<'a, TEO, S, D>
    where A: Scalar,
          S: DataMut<Elem = A> + DataClone,
          D: Dimension,
          TEO: TimeEvolution<D, Scalar = A>
{
    pub fn iterate(&mut self) {
        self.teo.iterate(&mut self.state, &mut self.buf);
    }
}

impl<'a, TEO, A, S, D> Iterator for TimeSeries<'a, TEO, S, D>
    where A: Scalar,
          S: DataMut<Elem = A> + DataClone,
          D: Dimension,
          TEO: TimeEvolution<D, Scalar = A>
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

impl<TEO> WithBuffer for NStep<TEO>
    where TEO: WithBuffer
{
    type Buffer = TEO::Buffer;
    fn new_buffer(&self) -> Self::Buffer {
        self.teo.new_buffer()
    }
}

impl<TEO, D> TimeEvolution<D> for NStep<TEO>
    where TEO: TimeEvolution<D>,
          D: Dimension
{
    type Scalar = TEO::Scalar;

    fn iterate<'a, S>(&self,
                      x: &'a mut ArrayBase<S, D>,
                      mut buf: &mut Self::Buffer)
                      -> &'a mut ArrayBase<S, D>
        where S: DataMut<Elem = TEO::Scalar>
    {
        for _ in 0..self.n {
            self.teo.iterate(x, buf);
        }
        x
    }
}
