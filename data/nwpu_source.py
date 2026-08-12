import numpy as np


def get_nwpu_images(data_path):
    """
    Load NWPU-RESISC45 as a list of (H, W, 3) uint8 numpy images - an
    alternative external "change-source" pool for draem_perlin(), in-domain
    (real aerial/satellite scenes, 256x256, 45 diverse scene classes) but
    out-of-dataset, as opposed to DTD's out-of-domain generic textures (see
    SyntheticCDDataModule's `change_source` option).

    Downloads via HuggingFace datasets on first use, cached under `data_path`.
    """
    from datasets import load_dataset

    ds = load_dataset("jonathan-roberts1/NWPU-RESISC45", split="train", cache_dir=str(data_path))
    return [np.array(row["image"].convert("RGB")) for row in ds]
