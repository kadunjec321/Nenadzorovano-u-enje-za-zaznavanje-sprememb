import numpy as np


def get_dtd_images(data_path):
    """
    Load the Describable Textures Dataset (DTD) as a list of (H, W, 3) uint8
    numpy images - an external, out-of-domain "change-source" texture pool
    for draem_perlin(), as an alternative to reusing the target dataset's own
    images (see SyntheticCDDataModule's `change_source` option).

    Downloads via torchvision on first use, cached under `data_path`.
    """
    from torchvision.datasets import DTD

    ds = DTD(root=str(data_path), split="train", download=True)
    return [np.array(img.convert("RGB")) for img, _ in ds]
