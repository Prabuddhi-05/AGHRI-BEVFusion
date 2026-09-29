"""AGHRI-specific data loading pipelines.

These loaders adapt data in memory. They never modify the converted dataset.
"""

from PIL import Image

from mmdet.datasets.builder import PIPELINES


@PIPELINES.register_module()
class LoadAGHRIMultiViewImageFromFiles:
    """Load AGHRI camera images as three-channel RGB PIL images.

    ZED PNGs may be stored as RGBA. The alpha channel is removed in memory so
    that the standard three-channel Swin backbone and ImageNet normalization
    can be used. Fisheye RGB images pass through the same explicit conversion.
    """

    def __init__(self, enforce_equal_size=True):
        self.enforce_equal_size = enforce_equal_size

    def __call__(self, results):
        filenames = results["image_paths"]
        if not filenames:
            raise ValueError("AGHRI sample has no camera image paths")

        images = []
        for filename in filenames:
            with Image.open(filename) as image:
                images.append(image.convert("RGB").copy())

        sizes = [image.size for image in images]
        if self.enforce_equal_size and len(set(sizes)) != 1:
            raise ValueError(
                "All cameras in one AGHRI mode must have the same native "
                f"image size; got {sizes} for {filenames}"
            )
        if any(image.mode != "RGB" for image in images):
            raise AssertionError("AGHRI RGB conversion did not produce RGB images")

        results["filename"] = filenames
        results["img"] = images
        results["img_shape"] = images[0].size
        results["ori_shape"] = images[0].size
        results["pad_shape"] = images[0].size
        results["scale_factor"] = 1.0
        return results

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"enforce_equal_size={self.enforce_equal_size})"
        )
