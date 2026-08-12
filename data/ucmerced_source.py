import numpy as np


def get_ucmerced_images(data_path):
    """
    Load UC Merced Land Use as a list of (H, W, 3) uint8 numpy images - an
    alternative external "change-source" pool for draem_perlin(). In-domain
    (real aerial scenes, already 256x256) with semantically concrete classes
    (buildings, forest, river, freeway, ...), narrower/more object-like than
    NWPU-RESISC45's broader scene classes - see SyntheticCDDataModule's
    `change_source` option.

    Downloads via HuggingFace datasets on first use, cached under `data_path`.
    """
    from datasets import load_dataset

    ds = load_dataset("blanchon/UC_Merced", split="train", cache_dir=str(data_path))
    return [np.array(row["image"].convert("RGB")) for row in ds]
