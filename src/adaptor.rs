
use ndarray::*;
use super::traits::*;

#[derive(new)]
pub struct TimeSeries<S, D, TEO>
    where S: DataMut,
          D: Dimension
{
    state: ArrayBase<S, D>,
    teo: TEO,
}

impl<S, D, TEO> TimeSeries<S, D, TEO>
    where S: DataMut + DataClone,
          D: Dimension,
          for<'b> &'b TEO: TimeEvolution<S, D>
{
    pub fn iterate(&mut self) {
        self.teo.iterate(&mut self.state);
    }
}

impl<'a, S, D, TEO> Iterator for &'a mut TimeSeries<S, D, TEO>
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
