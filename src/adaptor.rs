
use ndarray::*;
use super::traits::*;

#[derive(new)]
pub struct TimeSeries<'a, TEO, S, D>
    where S: DataMut,
          D: Dimension,
          TEO: 'a
{
    state: ArrayBase<S, D>,
    teo: &'a TEO,
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
