
use ndarray::*;
use super::traits::*;

pub struct TimeSeries<'a, TEO, S, D>
    where S: DataMut,
          D: Dimension,
          TEO: 'a,
          for<'b> &'b TEO: TimeEvolution<S, D>
{
    state: ArrayBase<S, D>,
    teo: &'a TEO,
}

pub fn time_series<'a, TEO, S, D>(x0: ArrayBase<S, D>, teo: &'a TEO) -> TimeSeries<'a, TEO, S, D>
    where S: DataMut,
          D: Dimension,
          TEO: 'a,
          for<'b> &'b TEO: TimeEvolution<S, D>
{
    TimeSeries {
        state: x0,
        teo: teo,
    }
}

impl<'a, TEO, S, D> TimeSeries<'a, TEO, S, D>
    where S: DataMut + DataClone,
          D: Dimension,
          for<'b> &'b TEO: TimeEvolution<S, D>
{
    pub fn iterate(&mut self) {
        self.teo.iterate(&mut self.state);
    }
}

impl<'a, TEO, S, D> Iterator for TimeSeries<'a, TEO, S, D>
    where S: DataMut + DataClone,
          D: Dimension,
          for<'b> &'b TEO: TimeEvolution<S, D>
{
    type Item = ArrayBase<S, D>;
    fn next(&mut self) -> Option<Self::Item> {
        self.iterate();
        Some(self.state.clone())
    }
}
