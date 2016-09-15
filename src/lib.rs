
extern crate ndarray;
extern crate num;

use ndarray::prelude::*;

#[inline(always)]
pub fn lorenz63(p: f64, b: f64, r: f64, mut v: Array<f64, Ix>) -> Array<f64, Ix> {
    let x = v[0];
    let y = v[1];
    let z = v[2];
    v[0] = p * (y - x);
    v[1] = x * (r - z) - y;
    v[2] = x * y - b * z;
    v
}

#[inline(always)]
pub fn roessler(a: f64, b: f64, c: f64, mut v: Array<f64, Ix>) -> Array<f64, Ix> {
    let x = v[0];
    let y = v[1];
    let z = v[2];
    v[0] = -y - z;
    v[1] = x + a * y;
    v[2] = b + x * z - c * z;
    v
}

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
    x + dt * (k1 + 2.0 * (k2 + k3) + l) / 6.0
}
