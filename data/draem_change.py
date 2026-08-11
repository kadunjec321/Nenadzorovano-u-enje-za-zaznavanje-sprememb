import numpy as np
from PIL import Image


def _fade(t):
    return 6 * t**5 - 15 * t**4 + 10 * t**3


def _perlin_noise(h, w, res, rng):
    """
    Vectorized 2D Perlin noise on an (h, w) grid, built from a (res+1, res+1)
    grid of random gradient vectors. Standard reference implementation
    (gradient noise + smoothstep interpolation), same approach DRAEM itself
    uses for its anomaly masks.
    """
    h_pad = ((h + res - 1) // res) * res
    w_pad = ((w + res - 1) // res) * res
    d = (h_pad // res, w_pad // res)

    grid = np.mgrid[0:res : 1 / d[0], 0:res : 1 / d[1]].transpose(1, 2, 0) % 1
    angles = 2 * np.pi * rng.random((res + 1, res + 1))
    gradients = np.dstack((np.cos(angles), np.sin(angles)))
    gradients = gradients.repeat(d[0], axis=0).repeat(d[1], axis=1)

    g00 = gradients[: -d[0], : -d[1]]
    g10 = gradients[d[0] :, : -d[1]]
    g01 = gradients[: -d[0], d[1] :]
    g11 = gradients[d[0] :, d[1] :]

    n00 = np.sum(grid * g00, axis=2)
    n10 = np.sum(np.dstack((grid[:, :, 0] - 1, grid[:, :, 1])) * g10, axis=2)
    n01 = np.sum(np.dstack((grid[:, :, 0], grid[:, :, 1] - 1)) * g01, axis=2)
    n11 = np.sum(np.dstack((grid[:, :, 0] - 1, grid[:, :, 1] - 1)) * g11, axis=2)

    t = _fade(grid)
    n0 = n00 * (1 - t[:, :, 0]) + t[:, :, 0] * n10
    n1 = n01 * (1 - t[:, :, 0]) + t[:, :, 0] * n11
    noise = np.sqrt(2) * ((1 - t[:, :, 1]) * n0 + t[:, :, 1] * n1)

    return noise[:h, :w]


def _perlin_mask(h, w, area_ratio, rng, res_choices=(2, 4, 8)):
    """
    Organic blob-shaped binary mask via thresholded Perlin noise. The
    threshold is picked as the exact quantile of the noise field needed to
    hit a randomly-sampled target area within `area_ratio` - no rejection
    sampling needed, since we know the noise distribution directly.
    """
    target_area = rng.uniform(*area_ratio)
    res = int(rng.choice(res_choices))
    noise = _perlin_noise(h, w, res, rng)
    threshold = np.quantile(noise, 1 - target_area)
    mask = (noise > threshold).astype(np.uint8) * 255
    return mask


def draem_perlin(image, source_image, area_ratio, rng=None, beta_range=(0.1, 1.0)):
    """
    DRAEM-like synthetic change: an organic Perlin-noise-shaped mask, with
    `source_image` content SOFTLY alpha-blended into `image` inside that mask
    (opacity beta ~ Uniform(*beta_range)) instead of a hard cutpaste/cutmix
    replace. Returns (imageA, imageB, mask), same contract as cutpaste()/
    cutmix() in data/synthetic_change.py.

    `source_image` is just another image from the same pool for now (no
    external texture dataset like DTD yet - that's a separate follow-up).
    """
    rng = rng or np.random.default_rng()
    h, w = image.shape[:2]
    mask = _perlin_mask(h, w, area_ratio, rng)

    if source_image.shape[:2] != (h, w):
        source_image = np.array(Image.fromarray(source_image).resize((w, h)))

    beta = rng.uniform(*beta_range)
    mask_f = (mask.astype(np.float32) / 255.0)[..., None]

    image_f = image.astype(np.float32)
    source_f = source_image.astype(np.float32)
    blended = image_f * (1 - mask_f) + (beta * source_f + (1 - beta) * image_f) * mask_f

    imageA = image.copy()
    imageB = np.clip(blended, 0, 255).astype(image.dtype)

    return imageA, imageB, mask
