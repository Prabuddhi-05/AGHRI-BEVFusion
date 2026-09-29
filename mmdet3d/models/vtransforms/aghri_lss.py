"""AGHRI adapter for selectable pinhole or distortion-aware LSS geometry."""

import torch

from mmdet3d.models.builder import VTRANSFORMS

from .lss import LSSTransform
from .camera_geometry import pixels_to_camera_rays

__all__ = ["AGHRILSSTransform"]


@VTRANSFORMS.register_module()
class AGHRILSSTransform(LSSTransform):
    """Use standard LSS features with optional raw-camera inverse projection.

    The locally modified BaseTransform passes a matrix dictionary to
    ``get_cam_feats``. Standard LSS does not use those matrices to construct
    image features. The default ``distortion_aware=False`` path calls the
    unchanged standard geometry and therefore preserves the working baseline.
    """

    def __init__(
        self,
        *args,
        distortion_aware=False,
        distortion_iterations=10,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.distortion_aware = bool(distortion_aware)
        self.distortion_iterations = int(distortion_iterations)
        if self.distortion_iterations <= 0:
            raise ValueError("distortion_iterations must be positive")

    def get_cam_feats(self, x, mats_dict=None):
        del mats_dict
        return super().get_cam_feats(x)

    def get_geometry(
        self,
        camera2lidar_rots,
        camera2lidar_trans,
        intrins,
        post_rots,
        post_trans,
        **kwargs,
    ):
        if not self.distortion_aware:
            return super().get_geometry(
                camera2lidar_rots,
                camera2lidar_trans,
                intrins,
                post_rots,
                post_trans,
                **kwargs,
            )

        model_codes = kwargs.get("camera_distortion_models")
        coefficients = kwargs.get("camera_distortion_coefficients")
        if model_codes is None or coefficients is None:
            raise ValueError(
                "distortion-aware AGHRI LSS requires camera distortion "
                "model codes and coefficients from the PKL metadata"
            )

        B, N, _ = camera2lidar_trans.shape
        points = self.frustum - post_trans.view(B, N, 1, 1, 1, 3)
        points = (
            torch.inverse(post_rots)
            .view(B, N, 1, 1, 1, 3, 3)
            .matmul(points.unsqueeze(-1))
            .squeeze(-1)
        )

        # ``points[..., :2]`` are now raw-image pixels. Preserve LSS's
        # camera-forward Z-depth convention by using rays (X/Z, Y/Z, 1), not
        # unit rays, before multiplying by the existing depth bins.
        rays = pixels_to_camera_rays(
            points[..., :2],
            intrins,
            model_codes,
            coefficients,
            iterations=self.distortion_iterations,
        )
        camera_points = rays * points[..., 2:3]
        points = (
            camera2lidar_rots.view(B, N, 1, 1, 1, 3, 3)
            .matmul(camera_points.unsqueeze(-1))
            .squeeze(-1)
        )
        points += camera2lidar_trans.view(B, N, 1, 1, 1, 3)

        if "extra_rots" in kwargs:
            points = (
                kwargs["extra_rots"]
                .view(B, 1, 1, 1, 1, 3, 3)
                .matmul(points.unsqueeze(-1))
                .squeeze(-1)
            )
        if "extra_trans" in kwargs:
            points += kwargs["extra_trans"].view(B, 1, 1, 1, 1, 3)
        return points
