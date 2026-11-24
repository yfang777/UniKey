import torch.nn.functional as F


RGB_IMAGE_SIZE = (224, 224)


def resize_rgb_tensor(image):
    """Resize RGB batches or sequences to the policy resolution before normalization."""
    if image.ndim == 4:
        return F.interpolate(
            image,
            size=RGB_IMAGE_SIZE,
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )

    batch_size, n_obs_steps, channels, height, width = image.shape
    resized = F.interpolate(
        image.reshape(batch_size * n_obs_steps, channels, height, width),
        size=RGB_IMAGE_SIZE,
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )
    return resized.reshape(batch_size, n_obs_steps, channels, *RGB_IMAGE_SIZE)


def resize_rgb_observation(batch, camera_keys):
    resized_images = {
        camera_key: resize_rgb_tensor(batch[camera_key])
        for camera_key in camera_keys
    }
    return {**batch, **resized_images}
