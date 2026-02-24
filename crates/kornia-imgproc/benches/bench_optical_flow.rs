use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use kornia_image::{Image, allocator::CpuAllocator};
use kornia_imgproc::optical_flow_pyr_lk::{calc_optical_flow_pyr_lk, PyrLKParams};

fn make_synthetic_pair(size: usize, dx: f32, dy: f32, n_points: usize) -> (Image<f32, 1, CpuAllocator>, Image<f32, 1, CpuAllocator>, Vec<[f32; 2]>) {
    let mut img1 = Image::<f32, 1, CpuAllocator>::from_size_val((size, size), 0.0, CpuAllocator);
    for y in 0..size {
        for x in 0..size {
            let cx = size as f32 / 2.0;
            let cy = size as f32 / 2.0;
            let r = 10.0;
            let dx_ = x as f32 - cx;
            let dy_ = y as f32 - cy;
            if (dx_ * dx_ + dy_ * dy_).sqrt() < r {
                *img1.get_mut([y, x, 0]) = 1.0;
            }
        }
    }
    let mut img2 = Image::<f32, 1, CpuAllocator>::from_size_val((size, size), 0.0, CpuAllocator);
    for y in 0..size {
        for x in 0..size {
            let cx = size as f32 / 2.0 + dx;
            let cy = size as f32 / 2.0 + dy;
            let r = 10.0;
            let dx_ = x as f32 - cx;
            let dy_ = y as f32 - cy;
            if (dx_ * dx_ + dy_ * dy_).sqrt() < r {
                *img2.get_mut([y, x, 0]) = 1.0;
            }
        }
    }
    let mut pts = Vec::with_capacity(n_points);
    let step = (size as f32 - 2.0 * r) / (n_points as f32).sqrt();
    let mut y = r + 1.0;
    while y < (size as f32 - r - 1.0) {
        let mut x = r + 1.0;
        while x < (size as f32 - r - 1.0) {
            pts.push([x, y]);
            if pts.len() >= n_points { break; }
            x += step;
        }
        if pts.len() >= n_points { break; }
        y += step;
    }
    (img1, img2, pts)
}

fn bench_lk(c: &mut Criterion) {
    let size = 128;
    let n_points = 1000;
    let dx = 3.5;
    let dy = -2.2;
    let (img1, img2, pts) = make_synthetic_pair(size, dx, dy, n_points);
    let params = PyrLKParams {
        win_size: 21,
        max_level: 3,
        max_iter: 30,
        epsilon: 1e-3,
        min_eigen_threshold: 1e-4,
    };
    c.bench_with_input(BenchmarkId::new("lk_synthetic", n_points), &pts, |b, pts| {
        b.iter(|| {
            let _ = calc_optical_flow_pyr_lk(&img1, &img2, pts, &params);
        });
    });
}

criterion_group!(benches, bench_lk);
criterion_main!(benches);
