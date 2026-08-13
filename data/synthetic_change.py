import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from data.draem_change import draem_perlin_multi

# Per-dataset calibration measured from each dataset's real train-split label
# masks (fraction of changed pixels per image, n=200 sample): `area_ratio`
# spans roughly p10-p90 (with margin toward the max) of real change area when
# a change IS present; `no_change_prob` matches the measured fraction of
# image pairs with near-zero (<0.5%) real change area.
#   clcd:   mean 4.3%, median 1.0%, p90 12.8%, max 44.3%, 44.5% near-zero
#   oscd96: mean 1.8%, median 0.4%, p90  4.5%, max 53.4%, 53.0% near-zero
#   gvlm:   mean 7.7%, median 0.0%, p90 22.6%, max  100%, 71.0% near-zero
DATASET_CALIBRATION = {
    "clcd": {"no_change_prob": 0.40, "area_ratio": (0.01, 0.20)},
    "oscd96": {"no_change_prob": 0.50, "area_ratio": (0.005, 0.10)},
    "gvlm": {"no_change_prob": 0.65, "area_ratio": (0.01, 0.40)},
}
DEFAULT_CALIBRATION = {"no_change_prob": 0.3, "area_ratio": (0.02, 0.15)}


def _random_rect(h, w, area_ratio, aspect_ratio, rng):
    area = rng.uniform(*area_ratio) * h * w
    ratio = np.exp(rng.uniform(np.log(aspect_ratio[0]), np.log(aspect_ratio[1])))
    cut_h = int(round(np.sqrt(area / ratio)))
    cut_w = int(round(np.sqrt(area * ratio)))
    cut_h = min(max(cut_h, 1), h)
    cut_w = min(max(cut_w, 1), w)
    return cut_h, cut_w


def cutpaste(image, area_ratio=(0.02, 0.15), aspect_ratio=(0.3, 3.3), rng=None):
    """
    Cut a random rectangular patch from `image` and paste it back at another
    location in the SAME image. Returns (imageA, imageB, mask) where mask
    marks the pasted region - imageA/before is unchanged, imageB/after has
    the pasted patch.
    """
    rng = rng or np.random.default_rng()
    h, w = image.shape[:2]
    cut_h, cut_w = _random_rect(h, w, area_ratio, aspect_ratio, rng)

    src_y = rng.integers(0, h - cut_h + 1)
    src_x = rng.integers(0, w - cut_w + 1)
    patch = image[src_y : src_y + cut_h, src_x : src_x + cut_w].copy()

    dst_y = rng.integers(0, h - cut_h + 1)
    dst_x = rng.integers(0, w - cut_w + 1)

    imageA = image.copy()
    imageB = image.copy()
    imageB[dst_y : dst_y + cut_h, dst_x : dst_x + cut_w] = patch

    mask = np.zeros((h, w), dtype=np.uint8)
    mask[dst_y : dst_y + cut_h, dst_x : dst_x + cut_w] = 255

    return imageA, imageB, mask


def cutmix(image, source_image, area_ratio=(0.02, 0.15), aspect_ratio=(0.3, 3.3), rng=None):
    """
    Cut a random rectangular patch from a DIFFERENT `source_image` and paste
    it into `image`. Returns (imageA, imageB, mask) same as cutpaste.
    """
    rng = rng or np.random.default_rng()
    h, w = image.shape[:2]
    sh, sw = source_image.shape[:2]
    cut_h, cut_w = _random_rect(h, w, area_ratio, aspect_ratio, rng)
    cut_h, cut_w = min(cut_h, sh), min(cut_w, sw)

    src_y = rng.integers(0, sh - cut_h + 1)
    src_x = rng.integers(0, sw - cut_w + 1)
    patch = source_image[src_y : src_y + cut_h, src_x : src_x + cut_w]
    if patch.shape[:2] != (cut_h, cut_w):
        # source pool spans datasets with different image sizes
        patch = np.array(Image.fromarray(patch).resize((cut_w, cut_h)))

    dst_y = rng.integers(0, h - cut_h + 1)
    dst_x = rng.integers(0, w - cut_w + 1)

    imageA = image.copy()
    imageB = image.copy()
    imageB[dst_y : dst_y + cut_h, dst_x : dst_x + cut_w] = patch

    mask = np.zeros((h, w), dtype=np.uint8)
    mask[dst_y : dst_y + cut_h, dst_x : dst_x + cut_w] = 255

    return imageA, imageB, mask


def _photometric_jitter(image, rng, brightness_range=(0.7, 1.3), contrast_range=(0.7, 1.3)):
    """
    Independent random brightness/contrast shift, applied separately to
    imageA and imageB (each call gets its own rng draws). Simulates the
    global lighting/color difference real bi-temporal pairs often have
    (season, sensor, time of day) that the model must learn to ignore -
    NOT the same thing as `color_match` in draem_change.py, which fixes a
    LOCAL patch-vs-surroundings mismatch instead. Done here (not via the
    shared albumentations pipeline in transforms.py) because that pipeline's
    `additional_targets` applies the SAME random draw to imageA and imageB,
    which would cancel out any A-vs-B difference.
    """
    img = image.astype(np.float32)
    brightness = rng.uniform(*brightness_range)
    contrast = rng.uniform(*contrast_range)
    mean = img.mean()
    img = (img - mean) * contrast + mean
    img = img * brightness
    return np.clip(img, 0, 255).astype(image.dtype)


class SyntheticChangeDataset(Dataset):
    """
    Generates (imageA, imageB, label) change-detection triples on the fly from
    a pool of single unlabeled images, via cutpaste, cutmix or draem (Perlin
    noise + soft blend). No real change labels are used - the label is
    exactly the pasted/blended region.

    Same __getitem__ return contract as CDDataset, so it's a drop-in
    replacement wherever a CDDataset is used (train.py, visualiser callback, ...).

    Pass `seed` to make generation deterministic per-index (e.g. for a val
    split that should stay stable across epochs); leave it None for train
    (fresh random pair every time an index is sampled).

    `no_change_prob` is the probability of emitting a genuine "nothing
    changed" pair (imageA == imageB, empty mask) instead of running
    cutpaste/cutmix - real change-detection datasets are mostly no-change
    pixels/images, so without this the model never sees that case at all.

    `source_images`, if given, is a SEPARATE pool used only as the "donor"
    content for cutmix/draem (e.g. DTD textures via data/dtd_source.py),
    instead of reusing `images` (the target dataset's own pool) for both
    roles. `soft_edges`, `color_match`, `beta_range` and `n_patches` (draem
    only) are passed straight through to draem_perlin_multi(). `n_patches`
    controls how many independent blobs (each with its own random donor
    image and opacity) make up the total change area - default 1 matches
    the original single-blob behavior. `photometric_aug`, if True, applies
    an independent random brightness/contrast shift to imageA and imageB
    (see `_photometric_jitter`) - a different, complementary variable from
    `color_match` (that one fixes a LOCAL patch/surroundings mismatch, this
    one adds a GLOBAL whole-image lighting difference the model must learn
    to ignore). All of these are optional, independently-testable
    variables, off by default (or n_patches=1 / beta_range unchanged).
    """

    def __init__(
        self,
        images,
        transform,
        method: str = "cutpaste",
        area_ratio: tuple[float, float] = (0.02, 0.15),
        aspect_ratio: tuple[float, float] = (0.3, 3.3),
        no_change_prob: float = 0.0,
        soft_edges: bool = False,
        color_match: bool = False,
        beta_range: tuple[float, float] = (0.1, 1.0),
        n_patches: int = 1,
        photometric_aug: bool = False,
        source_images=None,
        seed: int | None = None,
    ):
        if method not in ("cutpaste", "cutmix", "draem"):
            raise ValueError(
                f"Unknown synthetic method {method}, expected 'cutpaste', 'cutmix' or 'draem'"
            )

        self.images = images
        self.source_images = source_images if source_images is not None else images
        self.transform = transform
        self.method = method
        self.area_ratio = area_ratio
        self.aspect_ratio = aspect_ratio
        self.no_change_prob = no_change_prob
        self.soft_edges = soft_edges
        self.color_match = color_match
        self.beta_range = beta_range
        self.n_patches = n_patches
        self.photometric_aug = photometric_aug
        self.seed = seed

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        rng = np.random.default_rng(self.seed + index if self.seed is not None else None)

        image = np.asarray(self.images[index])
        if self.no_change_prob > 0 and rng.random() < self.no_change_prob:
            imageA, imageB = image.copy(), image.copy()
            label = np.zeros(image.shape[:2], dtype=np.uint8)
        elif self.method == "cutpaste":
            imageA, imageB, label = cutpaste(image, self.area_ratio, self.aspect_ratio, rng)
        elif self.method == "cutmix":
            other = np.asarray(self.source_images[rng.integers(0, len(self.source_images))])
            imageA, imageB, label = cutmix(image, other, self.area_ratio, self.aspect_ratio, rng)
        else:  # draem
            imageA, imageB, label = draem_perlin_multi(
                image,
                self.source_images,
                self.area_ratio,
                rng,
                beta_range=self.beta_range,
                soft_edges=self.soft_edges,
                color_match=self.color_match,
                n_patches=self.n_patches,
            )

        if self.photometric_aug:
            imageA = _photometric_jitter(imageA, rng)
            imageB = _photometric_jitter(imageB, rng)

        data = {"imageA": imageA, "imageB": imageB, "label": label, "img_idx": index}
        transformed = self.transform(data)
        transformed["imageA_unnorm"] = imageA
        transformed["imageB_unnorm"] = imageB
        transformed["img_idx"] = index

        return transformed
