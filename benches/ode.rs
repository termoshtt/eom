#[macro_use]
extern crate criterion;
extern crate eom;

use criterion::Criterion;
use eom::traits::*;
use eom::*;
use ndarray::arr1;

fn lorenz63_heun(c: &mut Criterion) {
    c.bench_function("Lorenz63 Heun 10000 steps", |b| {
        b.iter(|| {
            let dt = 0.01;
            let eom = ode::Lorenz63::default();
            let mut teo = explicit::Heun::new(eom, dt);
            let mut x = arr1(&[1.0, 0.0, 0.0]);
            for _ in 0..10000 {
                teo.iterate(&mut x);
            }
        })
    });
}

fn lorenz63_rk4(c: &mut Criterion) {
    c.bench_function("Lorenz63 RK4 10000 steps", |b| {
        b.iter(|| {
            let dt = 0.01;
            let eom = ode::Lorenz63::default();
            let mut teo = explicit::RK4::new(eom, dt);
            let mut x = arr1(&[1.0, 0.0, 0.0]);
            for _ in 0..10000 {
                teo.iterate(&mut x);
            }
        })
    });
}

criterion_group!(lorenz63, lorenz63_heun, lorenz63_rk4);
criterion_main!(lorenz63);
