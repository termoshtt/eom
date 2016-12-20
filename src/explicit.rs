
use ndarray::prelude::*;

pub fn euler<TEO, D: Dimension>(u: &TEO, dt: f64, x: Array<f64, D>) -> Array<f64, D>
    where TEO: Fn(Array<f64, D>) -> Array<f64, D>
{
    let y = x.clone();
    x + dt * u(y)
}

pub fn rk4<TEO, D: Dimension>(u: &TEO, dt: f64, x: Array<f64, D>) -> Array<f64, D>
    where TEO: Fn(Array<f64, D>) -> Array<f64, D>
{
    let mut l = x.clone();
    l = u(l);
    let k1 = l.clone();
    l = (0.5 * dt) * l + &x;
    l = u(l);
    let k2 = l.clone();
    l = (0.5 * dt) * l + &x;
    l = u(l);
    let k3 = l.clone();
    l = dt * l + &x;
    l = u(l);
    x + (dt / 6.0) * (k1 + 2.0 * (k2 + k3) + l)
}
