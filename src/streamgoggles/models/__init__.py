"""Neural network models and loss functions."""


def build_model(in_channels, image_size=None, kind="unet", **options):
    """The network named by ``kind``, built from its options.

    - ``"unet"`` (the default): `UNet`, one answer per pixel; ``options`` are
      its keyword arguments (``out_channels`` defaults to 1).
    - ``"hough"``: `HoughUNet`, one answer per line through the window;
      needs ``image_size`` (pixels on a side) and takes its keyword arguments.

    So a configuration chooses its model by options alone, and the per-pixel
    U-Net stays what it was when nothing else is asked.
    """
    if kind == "unet":
        from streamgoggles.models.unet import UNet

        return UNet(in_channels=in_channels, **{"out_channels": 1, **options})
    if kind == "hough":
        from streamgoggles.models.hough import HoughUNet

        if image_size is None:
            raise ValueError("a Hough model needs image_size")
        return HoughUNet(in_channels, image_size, **options)
    raise ValueError(f"unknown model kind {kind!r} (unet, hough)")
