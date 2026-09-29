"""Torch camera-ray geometry for raw AGHRI images.

The inverse projections return rays in ``(X/Z, Y/Z, 1)`` form.  This is
intentional: the LSS frustum bins in this repository represent camera-forward
Z depth, not Euclidean distance along a unit ray.
"""

import math

import torch


PINHOLE = 0
RATIONAL_POLYNOMIAL = 1
EQUIDISTANT = 2
MAX_DISTORTION_COEFFICIENTS = 8


def _safe_denominator(value, eps=1.0e-9):
    sign = torch.where(value < 0, -torch.ones_like(value), torch.ones_like(value))
    return torch.where(value.abs() < eps, sign * eps, value)


def inverse_rational_polynomial(distorted, coefficients, iterations=10):
    """Invert OpenCV's 8-term rational Brown-Conrady model with Newton steps.

    Args:
        distorted: Tensor with final dimension ``[x_d, y_d]``.
        coefficients: Broadcastable tensor in
            ``[k1, k2, p1, p2, k3, k4, k5, k6]`` order.
        iterations: Fixed Newton iteration count.
    """
    xd, yd = distorted.unbind(dim=-1)
    k1, k2, p1, p2, k3, k4, k5, k6 = coefficients.unbind(dim=-1)
    x = xd.clone()
    y = yd.clone()

    for _ in range(iterations):
        r2 = x * x + y * y
        r4 = r2 * r2
        r6 = r4 * r2

        numerator = 1 + k1 * r2 + k2 * r4 + k3 * r6
        denominator = 1 + k4 * r2 + k5 * r4 + k6 * r6
        denominator = _safe_denominator(denominator)
        radial = numerator / denominator

        tangent_x = 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
        tangent_y = p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
        residual_x = x * radial + tangent_x - xd
        residual_y = y * radial + tangent_y - yd

        dnum_dr2 = k1 + 2 * k2 * r2 + 3 * k3 * r4
        dden_dr2 = k4 + 2 * k5 * r2 + 3 * k6 * r4
        drad_dr2 = (
            dnum_dr2 * denominator - numerator * dden_dr2
        ) / (denominator * denominator)
        drad_dx = 2 * x * drad_dr2
        drad_dy = 2 * y * drad_dr2

        j00 = radial + x * drad_dx + 2 * p1 * y + 6 * p2 * x
        j01 = x * drad_dy + 2 * p1 * x + 2 * p2 * y
        j10 = y * drad_dx + 2 * p1 * x + 2 * p2 * y
        j11 = radial + y * drad_dy + 6 * p1 * y + 2 * p2 * x
        determinant = _safe_denominator(j00 * j11 - j01 * j10)

        step_x = (j11 * residual_x - j01 * residual_y) / determinant
        step_y = (-j10 * residual_x + j00 * residual_y) / determinant
        x = x - step_x
        y = y - step_y

    return torch.stack((x, y), dim=-1)


def inverse_equidistant(distorted, coefficients, iterations=10):
    """Invert OpenCV's four-term equidistant fisheye model.

    Rays at or beyond 90 degrees cannot be represented with positive Z-depth.
    Their angle is capped just below 90 degrees, sending their lifted samples
    far outside the finite AGHRI BEV grid instead of reflecting them behind the
    camera.
    """
    xd, yd = distorted.unbind(dim=-1)
    k1, k2, k3, k4 = coefficients[..., :4].unbind(dim=-1)
    theta_d = torch.sqrt(xd * xd + yd * yd)
    theta = theta_d.clone()

    for _ in range(iterations):
        theta2 = theta * theta
        theta4 = theta2 * theta2
        theta6 = theta4 * theta2
        theta8 = theta4 * theta4
        polynomial = 1 + k1 * theta2 + k2 * theta4 + k3 * theta6 + k4 * theta8
        residual = theta * polynomial - theta_d
        derivative = (
            1
            + 3 * k1 * theta2
            + 5 * k2 * theta4
            + 7 * k3 * theta6
            + 9 * k4 * theta8
        )
        theta = theta - residual / _safe_denominator(derivative)

    theta2 = theta * theta
    theta4 = theta2 * theta2
    theta6 = theta4 * theta2
    theta8 = theta4 * theta4
    final_residual = theta * (
        1 + k1 * theta2 + k2 * theta4 + k3 * theta6 + k4 * theta8
    ) - theta_d
    theta_limit = math.pi / 2 - 1.0e-4
    valid_positive_z = (
        torch.isfinite(theta)
        & (theta >= 0)
        & (theta < theta_limit)
        & (final_residual.abs() < 1.0e-5)
    )
    theta = torch.where(
        valid_positive_z,
        theta,
        torch.full_like(theta, theta_limit),
    )
    scale = torch.where(
        theta_d > 1.0e-9,
        torch.tan(theta) / _safe_denominator(theta_d),
        torch.ones_like(theta_d),
    )
    return torch.stack((xd * scale, yd * scale), dim=-1)


def pixels_to_camera_rays(
    pixels,
    intrinsics,
    model_codes,
    distortion_coefficients,
    iterations=10,
):
    """Convert raw-image pixels to camera rays with Z fixed to one.

    ``pixels`` has shape ``[B, N, ..., 2]``; calibration tensors start with
    ``[B, N]``. Model codes are 0=pinhole, 1=rational polynomial and
    2=equidistant.
    """
    if pixels.shape[-1] != 2:
        raise ValueError("pixels must have final dimension 2")
    if intrinsics.shape[-2:] != (3, 3):
        raise ValueError("intrinsics must end in 3x3")
    if distortion_coefficients.shape[-1] < MAX_DISTORTION_COEFFICIENTS:
        raise ValueError("distortion coefficients must be padded to eight values")

    spatial_dims = pixels.ndim - 3
    expand = (1,) * spatial_dims
    k = intrinsics.view(*intrinsics.shape[:2], *expand, 3, 3)
    coeffs = distortion_coefficients.view(
        *distortion_coefficients.shape[:2], *expand, distortion_coefficients.shape[-1]
    )

    ones = torch.ones_like(pixels[..., :1])
    homogeneous = torch.cat((pixels, ones), dim=-1)
    normalized_h = torch.linalg.inv(k).matmul(homogeneous.unsqueeze(-1)).squeeze(-1)
    pinhole_xy = normalized_h[..., :2] / _safe_denominator(normalized_h[..., 2:3])

    rational_xy = inverse_rational_polynomial(
        pinhole_xy, coeffs[..., :8], iterations=iterations
    )

    # OpenCV's fisheye API represents skew through a separate alpha argument;
    # AGHRI's validated cv2.fisheye projection uses alpha=0. Reproduce that
    # convention here rather than interpreting K[0, 1] as pinhole skew.
    fx = k[..., 0, 0]
    fy = k[..., 1, 1]
    cx = k[..., 0, 2]
    cy = k[..., 1, 2]
    equidistant_distorted = torch.stack(
        ((pixels[..., 0] - cx) / fx, (pixels[..., 1] - cy) / fy), dim=-1
    )
    equidistant_xy = inverse_equidistant(
        equidistant_distorted, coeffs[..., :4], iterations=iterations
    )

    codes = model_codes.view(*model_codes.shape[:2], *expand, 1)
    xy = torch.where(codes == RATIONAL_POLYNOMIAL, rational_xy, pinhole_xy)
    xy = torch.where(codes == EQUIDISTANT, equidistant_xy, xy)
    return torch.cat((xy, torch.ones_like(xy[..., :1])), dim=-1)
