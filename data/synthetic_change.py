import numpy as np
from PIL import Image
from torch.utils.data import Dataset


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


class SyntheticChangeDataset(Dataset):
    """
    Generates (imageA, imageB, label) change-detection triples on the fly from
    a pool of single unlabeled images, via cutpaste or cutmix. No real change
    labels are used - the label is exactly the pasted region.

    Same __getitem__ return contract as CDDataset, so it's a drop-in
    replacement wherever a CDDataset is used (train.py, visualiser callback, ...).

    Pass `seed` to make generation deterministic per-index (e.g. for a val
    split that should stay stable across epochs); leave it None for train
    (fresh random pair every time an index is sampled).
    """

    def __init__(
        self,
        images,
        transform,
        method: str = "cutpaste",
        area_ratio: tuple[float, float] = (0.02, 0.15),
        aspect_ratio: tuple[float, float] = (0.3, 3.3),
        seed: int | None = None,
    ):
        if method not in ("cutpaste", "cutmix"):
            raise ValueError(f"Unknown synthetic method {method}, expected 'cutpaste' or 'cutmix'")

        self.images = images
        self.transform = transform
        self.method = method
        self.area_ratio = area_ratio
        self.aspect_ratio = aspect_ratio
        self.seed = seed

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        rng = np.random.default_rng(self.seed + index if self.seed is not None else None)

        image = np.asarray(self.images[index])
        if self.method == "cutpaste":
            imageA, imageB, label = cutpaste(image, self.area_ratio, self.aspect_ratio, rng)
        else:
            other = np.asarray(self.images[rng.integers(0, len(self.images))])
            imageA, imageB, label = cutmix(image, other, self.area_ratio, self.aspect_ratio, rng)

        data = {"imageA": imageA, "imageB": imageB, "label": label, "img_idx": index}
        transformed = self.transform(data)
        transformed["imageA_unnorm"] = imageA
        transformed["imageB_unnorm"] = imageB
        transformed["img_idx"] = index

        return transformed
