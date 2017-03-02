
use super::traits::TimeEvolution;
use ndarray::*;

/// Linear ODE with diagonalized matrix (exactly solvable)
pub struct Diagonal {
    diag: Vec<f64>,
    dt: f64,
}

impl Diagonal {
    pub fn new<Iter>(diag_of_matrix: Iter, dt: f64) -> Self
        where Iter: Iterator<Item = f64>
    {
        Diagonal {
            diag: diag_of_matrix.map(|x| (x * dt).exp())
                .collect(),
            dt: dt,
        }
    }
}

impl TimeEvolution<Ix1> for Diagonal {
    fn iterate(&self, mut x: RcArray<f64, Ix1>) -> RcArray<f64, Ix1> {
        for (val, d) in x.iter_mut().zip(self.diag.iter()) {
            *val *= *d;
        }
        x
    }
    fn get_dt(&self) -> f64 {
        self.dt
    }
}
