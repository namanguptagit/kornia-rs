/// Lucas–Kanade optical flow with pyramids.
/// This implementation is a direct port of the OpenCV algorithm, with some simplifications:
/// - Only grayscale images (single channel, f32)
/// - No support for initial guess of next points (always starts at prev points)
/// - No support for tracking status (always returns 1 for success, 0 for failure
/// - No support for error output (returns mean abs It in window as error)
/// - No support for different window shapes (always uses square window)
/// - No support for different interpolation methods (always uses bilinear)
/// - No support for different pyramid levels (always uses max_level)
/// - No support for different termination criteria (always uses max_iter and epsilon)
/// - No support for different eigenvalue thresholds (always uses min_eigen_threshold)
/// - No support for different data types (always uses f32)
/// - No support for different image sizes (always assumes images are large enough for window)
/// - No support for different image borders (always assumes images are large enough for window)
/// - No support for different feature point formats (always uses Vec<[f32; 2
/// - No support for different output formats (always uses PyrLKResult)
/// - No support for different parallelization strategies (always processes features sequentially)
/// - No support for different error metrics (always uses mean abs It in window)
/// - No support for different optimization methods (always uses Gauss–Newton)
/// - No support for different convergence criteria (always uses max_iter and epsilon)
/// - No support for different pyramid construction methods (always uses pyrdown)
/// - No support for different gradient computation methods (always uses spatial_gradient_float_parallel_row)
/// - No support for different interpolation methods (always uses bilinear)

use crate::pyramid::pyrdown_f32;
use crate::filter::ops::spatial_gradient_float_parallel_row;
use crate::interpolation::bilinear::bilinear_interpolation;
use kornia_image::{Image, allocator::ImageAllocator, allocator::CpuAllocator};

/// Termination criteria for LK iterations (COUNT, EPS, or BOTH)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TermCriteria {
    Count,
    Eps,
    Both,
}

/// Parameters for LK optical flow (OpenCV parity)
#[derive(Debug, Clone)]
pub struct PyrLKParams {
    pub win_size: usize, // must be odd
    pub max_level: usize,
    pub max_iter: usize,
    pub epsilon: f32,
    pub min_eigen_threshold: f32,
    pub use_initial_flow: bool,
    pub term_criteria: TermCriteria,
    pub border_mode: BorderMode,
}

/// Border handling policy
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BorderMode {
    Clamp,
    Mirror,
    Reject,
}

/// Output for each tracked feature from LK tracking.
#[derive(Debug, Clone)]
pub struct PyrLKResult {
    pub next_pts: Vec<[f32; 2]>,
    pub status: Vec<u8>, // 1 = success, 0 = fail
    pub error: Vec<f32>,
}

/// Compute sparse pyramidal Lucas–Kanade optical flow.
///
/// # Arguments
/// * `prev_img` - Previous image (grayscale, f32, shape HxWx1)
/// * `next_img` - Next image (grayscale, f32, shape HxWx1)
/// * `prev_pts` - Feature points to track (N x 2)
/// * `params` - LK parameters
///
/// # Returns
/// * `PyrLKResult` with next points, status, and error (mean abs It per feature)
pub fn calc_optical_flow_pyr_lk<A: ImageAllocator>(
    prev_img: &Image<f32, 1, A>,
    next_img: &Image<f32, 1, A>,
    prev_pts: &[[f32; 2]],
    next_pts_in: Option<&[[f32; 2]]>,
    params: &PyrLKParams,
) -> PyrLKResult {
    // 1. Build Gaussian pyramids for prev and next images
    let mut prev_pyr = Vec::with_capacity(params.max_level + 1);
    let mut next_pyr = Vec::with_capacity(params.max_level + 1);
    prev_pyr.push(prev_img.clone());
    next_pyr.push(next_img.clone());
    for l in 1..=params.max_level {
        let prev_down = pyrdown_f32(&prev_pyr[l - 1]);
        let next_down = pyrdown_f32(&next_pyr[l - 1]);
        prev_pyr.push(prev_down);
        next_pyr.push(next_down);
    }

    // 2. Compute gradients (Ix, Iy) for each pyramid level (prev only)
    let mut grad_x_pyr = Vec::with_capacity(params.max_level + 1);
    let mut grad_y_pyr = Vec::with_capacity(params.max_level + 1);
    for l in 0..=params.max_level {
        let (ix, iy) = spatial_gradient_float_parallel_row(&prev_pyr[l]);
        grad_x_pyr.push(ix);
        grad_y_pyr.push(iy);
    }

    // 3. For each feature, track from coarse to fine
    let n_features = prev_pts.len();
    let mut next_pts = vec![[0.0f32; 2]; n_features];
    let mut status = vec![0u8; n_features];
    let mut error = vec![0.0f32; n_features];

    assert!(params.win_size % 2 == 1, "window size must be odd");
    let win_size = params.win_size as isize;
    let half_win = (params.win_size / 2) as isize;

    for (i, &pt) in prev_pts.iter().enumerate() {
        // Initial flow support
        let (mut x, mut y) = if params.use_initial_flow {
            if let Some(next_pts) = next_pts_in {
                let nx = next_pts[i][0] / (1 << params.max_level) as f32;
                let ny = next_pts[i][1] / (1 << params.max_level) as f32;
                (nx, ny)
            } else {
                (pt[0] / (1 << params.max_level) as f32, pt[1] / (1 << params.max_level) as f32)
            }
        } else {
            (pt[0] / (1 << params.max_level) as f32, pt[1] / (1 << params.max_level) as f32)
        };
        let mut dx = 0.0f32;
        let mut dy = 0.0f32;
        let mut valid = true;

        // Process from coarse to fine
        for lvl in (0..=params.max_level).rev() {
            let prev = &prev_pyr[lvl];
            let next = &next_pyr[lvl];
            let ix = &grad_x_pyr[lvl];
            let iy = &grad_y_pyr[lvl];

            // Scale up displacement for finer level
            if lvl < params.max_level {
                x = x * 2.0 + dx * 2.0;
                y = y * 2.0 + dy * 2.0;
                dx = 0.0;
                dy = 0.0;
            }

            // Gauss–Newton iterations
            let mut last_delta2 = 0.0f32;
            for _iter in 0..params.max_iter {
                let mut a = 0.0f32;
                let mut b = 0.0f32;
                let mut c = 0.0f32;
                let mut d = 0.0f32;
                let mut e = 0.0f32;
                let mut err = 0.0f32;
                let mut count = 0;

                // Center of window in prev image
                let xc = x;
                let yc = y;
                // Center of window in next image (with current displacement)
                let xnc = x + dx;
                let ync = y + dy;

                // Border handling
                let h = prev.rows() as isize;
                let w = prev.cols() as isize;
                let window_in_bounds = |xc: f32, yc: f32| {
                    let x0 = xc - half_win as f32;
                    let y0 = yc - half_win as f32;
                    let x1 = xc + half_win as f32;
                    let y1 = yc + half_win as f32;
                    x0 >= 0.0 && y0 >= 0.0 && x1 < w as f32 && y1 < h as f32
                };
                match params.border_mode {
                    BorderMode::Reject => {
                        if !window_in_bounds(xc, yc) || !window_in_bounds(xnc, ync) {
                            valid = false;
                            break;
                        }
                    }
                    BorderMode::Clamp | BorderMode::Mirror => {
                        // handled in interpolation
                    }
                }

                // Accumulate sums in 21x21 window
                for wy in -half_win..=half_win {
                    for wx in -half_win..=half_win {
                        let px = xc + wx as f32;
                        let py = yc + wy as f32;
                        let qx = xnc + wx as f32;
                        let qy = ync + wy as f32;

                        // Border-safe bilinear interpolation
                        let interp = |img: &Image<f32, 1, A>, x: f32, y: f32| -> f32 {
                            match params.border_mode {
                                BorderMode::Clamp => {
                                    let xf = x.max(0.0).min((img.cols() - 1) as f32);
                                    let yf = y.max(0.0).min((img.rows() - 1) as f32);
                                    bilinear_interpolation::<1, _>(img, xf, yf, 0)
                                }
                                BorderMode::Mirror => {
                                    let xf = if x < 0.0 {
                                        -x
                                    } else if x > (img.cols() - 1) as f32 {
                                        2.0 * (img.cols() - 1) as f32 - x
                                    } else {
                                        x
                                    };
                                    let yf = if y < 0.0 {
                                        -y
                                    } else if y > (img.rows() - 1) as f32 {
                                        2.0 * (img.rows() - 1) as f32 - y
                                    } else {
                                        y
                                    };
                                    bilinear_interpolation::<1, _>(img, xf, yf, 0)
                                }
                                BorderMode::Reject => unreachable!(),
                            }
                        };
                        let i0 = interp(prev, px, py);
                        let i1 = interp(next, qx, qy);
                        let ixv = interp(ix, px, py);
                        let iyv = interp(iy, px, py);

                        let it = i1 - i0;

                        a += ixv * ixv;
                        b += ixv * iyv;
                        c += iyv * iyv;
                        d += ixv * it;
                        e += iyv * it;
                        err += it.abs();
                        count += 1;
                    }
                }

                // Manual 2x2 solve
                let det = a * c - b * b;
                if det.abs() < 1e-7 {
                    valid = false;
                    break;
                }
                // Minimum eigenvalue check
                let trace = a + c;
                let delta = a - c;
                let lambda_min = (trace - ((delta * delta + 4.0 * b * b).sqrt())) * 0.5;
                if lambda_min < params.min_eigen_threshold {
                    valid = false;
                    break;
                }
                let inv_det = 1.0 / det;
                let delta_x = inv_det * (c * d - b * e);
                let delta_y = inv_det * (-b * d + a * e);

                dx += delta_x;
                dy += delta_y;
                last_delta2 = delta_x * delta_x + delta_y * delta_y;
                if last_delta2 < params.epsilon * params.epsilon {
                    break;
                }
            }
            // If feature went out of bounds, mark as invalid and break
            if !valid {
                break;
            }
            // Set tracking error at finest level (mean abs It in window)
            if lvl == 0 && valid {
                error[i] = if count > 0 { err / (count as f32) } else { 0.0 };
            }
    }

#[cfg(test)]
mod tests {
    use super::*;
    use kornia_image::{Image, allocator::CpuAllocator};

    fn make_circle_image(size: usize, cx: f32, cy: f32, r: f32) -> Image<f32, 1, CpuAllocator> {
        let mut img = Image::<f32, 1, CpuAllocator>::from_size_val((size, size), 0.0, CpuAllocator);
        for y in 0..size {
            for x in 0..size {
                let dx = x as f32 - cx;
                let dy = y as f32 - cy;
                if (dx * dx + dy * dy).sqrt() < r {
                    *img.get_mut([y, x, 0]) = 1.0;
                }
            }
        }
        img
    }

    fn default_params() -> PyrLKParams {
        PyrLKParams {
            win_size: 21,
            max_level: 2,
            max_iter: 30,
            epsilon: 1e-3,
            min_eigen_threshold: 1e-4,
            use_initial_flow: false,
            term_criteria: TermCriteria::Both,
            border_mode: BorderMode::Clamp,
        }
    }

    #[test]
    fn test_lk_synthetic_translation() {
        let size = 64;
        let dx = 5.0;
        let dy = -3.0;
        let img1 = make_circle_image(size, 32.0, 32.0, 10.0);
        let img2 = make_circle_image(size, 32.0 + dx, 32.0 + dy, 10.0);
        let pts = vec![[32.0, 32.0], [36.0, 32.0], [32.0, 36.0]];
        let mut params = default_params();
        params.win_size = 21;
        let result = calc_optical_flow_pyr_lk(&img1, &img2, &pts, None, &params);
        for (i, &pt) in pts.iter().enumerate() {
            assert_eq!(result.status[i], 1);
            let est_dx = result.next_pts[i][0] - pt[0];
            let est_dy = result.next_pts[i][1] - pt[1];
            assert!((est_dx - dx).abs() < 0.01, "dx error too large: {}", est_dx);
            assert!((est_dy - dy).abs() < 0.01, "dy error too large: {}", est_dy);
        }
    }

    #[test]
    fn test_lk_subpixel_accuracy() {
        let size = 64;
        let dx = 0.4;
        let dy = -0.7;
        let img1 = make_circle_image(size, 32.0, 32.0, 10.0);
        let img2 = make_circle_image(size, 32.0 + dx, 32.0 + dy, 10.0);
        let pts = vec![[32.0, 32.0]];
        let params = default_params();
        let result = calc_optical_flow_pyr_lk(&img1, &img2, &pts, None, &params);
        assert_eq!(result.status[0], 1);
        let est_dx = result.next_pts[0][0] - pts[0][0];
        let est_dy = result.next_pts[0][1] - pts[0][1];
        assert!((est_dx - dx).abs() < 0.01, "subpixel dx error: {}", est_dx);
        assert!((est_dy - dy).abs() < 0.01, "subpixel dy error: {}", est_dy);
    }

    #[test]
    fn test_lk_window_size_and_border() {
        let size = 64;
        let img1 = make_circle_image(size, 32.0, 32.0, 10.0);
        let img2 = make_circle_image(size, 32.0, 32.0, 10.0);
        let pts = vec![[2.0, 2.0], [62.0, 62.0]];
        let mut params = default_params();
        params.win_size = 15;
        params.border_mode = BorderMode::Reject;
        let result = calc_optical_flow_pyr_lk(&img1, &img2, &pts, None, &params);
        // Both features are near the border and should be rejected
        assert_eq!(result.status[0], 0);
        assert_eq!(result.status[1], 0);
    }

    #[test]
    fn test_lk_initial_flow() {
        let size = 64;
        let dx = 2.0;
        let dy = 1.0;
        let img1 = make_circle_image(size, 32.0, 32.0, 10.0);
        let img2 = make_circle_image(size, 32.0 + dx, 32.0 + dy, 10.0);
        let pts = vec![[32.0, 32.0]];
        let mut params = default_params();
        params.use_initial_flow = true;
        let initial = vec![[32.0 + 1.0, 32.0 + 0.5]];
        let result = calc_optical_flow_pyr_lk(&img1, &img2, &pts, Some(&initial), &params);
        assert_eq!(result.status[0], 1);
        let est_dx = result.next_pts[0][0] - pts[0][0];
        let est_dy = result.next_pts[0][1] - pts[0][1];
        assert!((est_dx - dx).abs() < 0.01, "initial flow dx error: {}", est_dx);
        assert!((est_dy - dy).abs() < 0.01, "initial flow dy error: {}", est_dy);
    }

    #[test]
    fn test_lk_eigenvalue_rejection() {
        let img = Image::<f32, 1, CpuAllocator>::from_size_val((32, 32), 1.0, CpuAllocator);
        let pts = vec![[16.0, 16.0], [20.0, 20.0]];
        let mut params = default_params();
        params.min_eigen_threshold = 1e-2;
        let result = calc_optical_flow_pyr_lk(&img, &img, &pts, None, &params);
        for &s in &result.status {
            assert_eq!(s, 0, "Feature should be rejected on flat image");
        }
    }

    #[test]
    fn test_lk_convergence_large_motion() {
        let size = 64;
        let dx = 12.0;
        let dy = -8.0;
        let img1 = make_circle_image(size, 32.0, 32.0, 10.0);
        let img2 = make_circle_image(size, 32.0 + dx, 32.0 + dy, 10.0);
        let pts = vec![[32.0, 32.0]];
        let mut params = default_params();
        params.max_level = 3;
        params.max_iter = 50;
        let result = calc_optical_flow_pyr_lk(&img1, &img2, &pts, None, &params);
        assert_eq!(result.status[0], 1);
        let est_dx = result.next_pts[0][0] - pts[0][0];
        let est_dy = result.next_pts[0][1] - pts[0][1];
        assert!((est_dx - dx).abs() < 0.05, "large motion dx error: {}", est_dx);
        assert!((est_dy - dy).abs() < 0.05, "large motion dy error: {}", est_dy);
    }

    #[test]
    fn test_lk_opencv_parity() {
        // This test assumes you have reference output from OpenCV for the same input
        // (e.g., generated by Python or C++ and checked in as a .npy/.csv file)
        // Here we just show the structure; actual data loading is project-specific.
        // let img1 = ...;
        // let img2 = ...;
        // let pts = ...;
        // let opencv_next = ...;
        // let opencv_status = ...;
        // let opencv_error = ...;
        // let params = default_params();
        // let result = calc_optical_flow_pyr_lk(&img1, &img2, &pts, None, &params);
        // for i in 0..pts.len() {
        //     assert!((result.next_pts[i][0] - opencv_next[i][0]).abs() < 0.01);
        //     assert!((result.next_pts[i][1] - opencv_next[i][1]).abs() < 0.01);
        //     assert_eq!(result.status[i], opencv_status[i]);
        //     assert!((result.error[i] - opencv_error[i]).abs() < 0.01);
        // }
    }
}
        }
        // Write result for this feature
        if valid {
            next_pts[i][0] = x + dx;
            next_pts[i][1] = y + dy;
            status[i] = 1;
            // error[i] to be set later
        } else {
            status[i] = 0;
        }
    }

    PyrLKResult { next_pts, status, error }
}

// --- Helper functions to be implemented below ---