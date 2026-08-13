import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter


def _fade(t):
    return 6 * t**5 - 15 * t**4 + 10 * t**3


def _perlin_noise_raw(h, w, res, rng):
    """
    Vectorized 2D Perlin noise on an (h, w) grid, built from a (res+1, res+1)
    grid of random gradient vectors. Standard reference implementation
    (gradient noise + smoothstep interpolation), same approach DRAEM itself
    uses for its anomaly masks.
    """
    res = max(1, min(res, min(h, w)))
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


def _perlin_noise(h, w, res, rng, angle=0.0):
    """
    Single-octave Perlin noise, optionally sampled on a grid rotated by
    `angle` radians - breaks the axis-aligned "grain" a fixed-orientation
    grid produces, so blob shapes don't all share the same bias.
    """
    if angle == 0.0:
        return _perlin_noise_raw(h, w, res, rng)

    # generate on a padded canvas so the post-rotation crop has no empty corners
    diag = int(np.ceil(np.hypot(h, w))) + 1
    noise = _perlin_noise_raw(diag, diag, res, rng)
    rotated = Image.fromarray(noise.astype(np.float32), mode="F").rotate(
        np.degrees(angle), resample=Image.BILINEAR
    )
    rotated = np.asarray(rotated)
    y0, x0 = (diag - h) // 2, (diag - w) // 2
    return rotated[y0 : y0 + h, x0 : x0 + w]


def _fractal_perlin_noise(h, w, rng, base_res_range=(2, 16), octaves=(1, 4), persistence=0.5, lacunarity=2.0):
    """
    Fractal (multi-octave) Perlin noise: sums several single-octave layers at
    doubling frequency and halving amplitude, all sampled on the SAME random
    grid rotation. A randomized base frequency, octave count and rotation
    together give much more varied blob shapes/scales than a single fixed-res
    layer - avoids the model latching onto "one typical blob shape" as a
    shortcut instead of actually comparing image content.
    """
    angle = rng.uniform(0, 2 * np.pi)
    res = int(rng.integers(base_res_range[0], base_res_range[1] + 1))
    n_octaves = int(rng.integers(octaves[0], octaves[1] + 1))

    noise = np.zeros((h, w), dtype=np.float32)
    amplitude, total_amplitude = 1.0, 0.0
    for _ in range(n_octaves):
        noise += amplitude * _perlin_noise(h, w, res, rng, angle=angle)
        total_amplitude += amplitude
        amplitude *= persistence
        res = int(round(res * lacunarity))

    return noise / total_amplitude


def _perlin_mask(
    h, w, area_ratio, rng, base_res_range=(2, 16), octaves=(1, 4), soft_edges=False, blur_sigma_range=(2.0, 6.0)
):
    """
    Organic blob-shaped mask via thresholded fractal Perlin noise. The
    threshold is picked as the exact quantile of the noise field needed to
    hit a randomly-sampled target area within `area_ratio` - no rejection
    sampling needed, since we know the noise distribution directly.

    By default the mask is hard-edged (binary 0/255), same as DRAEM's own
    masks. `soft_edges=True` Gaussian-blurs it into a gradual 0-255 ramp at
    the boundary instead of a sharp cutoff - an optional, separately testable
    variable, not applied automatically.
    """
    target_area = rng.uniform(*area_ratio)
    noise = _fractal_perlin_noise(h, w, rng, base_res_range=base_res_range, octaves=octaves)
    threshold = np.quantile(noise, 1 - target_area)
    mask = (noise > threshold).astype(np.float32) * 255

    if soft_edges:
        sigma = rng.uniform(*blur_sigma_range)
        mask = gaussian_filter(mask, sigma=sigma)

    return mask.astype(np.uint8)


def _match_color_stats(source, dest, weights):
    """
    Linearly rescale `source`'s per-channel mean/std to match `dest`'s,
    both measured only over the region given by `weights` (a soft 0-1 mask).
    Addresses an easy shortcut cue: a patch pulled from a differently-lit
    source image looks jarringly different from its LOCAL destination
    surroundings, unlike real changes which stay within the scene's own
    lighting/color range.
    """
    w = weights[..., None]
    total = w.sum()
    if total < 1.0:
        return source

    src_mean = (source * w).sum(axis=(0, 1)) / total
    dst_mean = (dest * w).sum(axis=(0, 1)) / total
    src_std = np.sqrt((w * (source - src_mean) ** 2).sum(axis=(0, 1)) / total) + 1e-6
    dst_std = np.sqrt((w * (dest - dst_mean) ** 2).sum(axis=(0, 1)) / total) + 1e-6

    matched = (source - src_mean) / src_std * dst_std + dst_mean
    return np.clip(matched, 0, 255)


def draem_perlin(
    image,
    source_image,
    area_ratio,
    rng=None,
    beta_range=(0.1, 1.0),
    soft_edges=False,
    color_match=False,
):
    """
    DRAEM-like synthetic change: an organic Perlin-noise-shaped mask, with
    `source_image` content SOFTLY alpha-blended into `image` inside that mask
    (opacity beta ~ Uniform(*beta_range)) instead of a hard cutpaste/cutmix
    replace. Returns (imageA, imageB, mask), same contract as cutpaste()/
    cutmix() in data/synthetic_change.py.

    `source_image` is another image from a pool - either the same target
    dataset's own images, or an external texture pool like DTD (see
    data/dtd_source.py and SyntheticCDDataModule's `change_source` option).

    `soft_edges=True` blurs the mask boundary (see `_perlin_mask`) instead of
    a hard cutoff. `color_match=True` rescales the source patch's local
    color/brightness statistics (within the mask) to match the destination's,
    before blending. Both are independent, optional variables, off by default.
    """
    rng = rng or np.random.default_rng()
    h, w = image.shape[:2]
    mask = _perlin_mask(h, w, area_ratio, rng, soft_edges=soft_edges)

    if source_image.shape[:2] != (h, w):
        source_image = np.array(Image.fromarray(source_image).resize((w, h)))

    beta = rng.uniform(*beta_range)
    mask_f = (mask.astype(np.float32) / 255.0)[..., None]

    image_f = image.astype(np.float32)
    source_f = source_image.astype(np.float32)

    if color_match:
        source_f = _match_color_stats(source_f, image_f, mask_f[..., 0])

    blended = image_f * (1 - mask_f) + (beta * source_f + (1 - beta) * image_f) * mask_f

    imageA = image.copy()
    imageB = np.clip(blended, 0, 255).astype(image.dtype)

    return imageA, imageB, mask


def draem_perlin_multi(
    image,
    source_pool,
    area_ratio,
    rng=None,
    beta_range=(0.1, 1.0),
    soft_edges=False,
    color_match=False,
    n_patches=1,
):
    """
    Multi-instance variant of draem_perlin(): splits the total change area
    across `n_patches` INDEPENDENT blobs, each drawing its own random donor
    image from `source_pool` and its own random beta - instead of one blob
    with one shared texture/opacity for the whole change. Meant to reduce
    the risk of the model latching onto "one specific recurring texture" as
    a shortcut, since no two patches in the same pair need look alike.

    n_patches=1 reduces to the same single-blob behavior as draem_perlin(),
    just sourced from a pool instead of one pre-picked image.
    """
    rng = rng or np.random.default_rng()
    h, w = image.shape[:2]
    image_f = image.astype(np.float32)
    blended = image_f.copy()
    combined_mask = np.zeros((h, w), dtype=np.float32)

    total_area = rng.uniform(*area_ratio)
    shares = rng.dirichlet(np.ones(n_patches)) * total_area

    for share in shares:
        mask = _perlin_mask(h, w, (share, share), rng, soft_edges=soft_edges)
        mask_f = mask.astype(np.float32) / 255.0

        source_image = source_pool[rng.integers(0, len(source_pool))]
        source_image = np.asarray(source_image)
        if source_image.shape[:2] != (h, w):
            source_image = np.array(Image.fromarray(source_image).resize((w, h)))
        source_f = source_image.astype(np.float32)

        if color_match:
            source_f = _match_color_stats(source_f, blended, mask_f)

        beta = rng.uniform(*beta_range)
        mask_f3 = mask_f[..., None]
        blended = blended * (1 - mask_f3) + (beta * source_f + (1 - beta) * blended) * mask_f3
        combined_mask = np.maximum(combined_mask, mask.astype(np.float32))

    imageA = image.copy()
    imageB = np.clip(blended, 0, 255).astype(image.dtype)

    return imageA, imageB, combined_mask.astype(np.uint8)
