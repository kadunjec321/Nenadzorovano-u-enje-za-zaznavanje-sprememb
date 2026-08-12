import numpy as np


def get_minc_images(data_path):
    """
    Load MINC-2500 (Materials in Context) as a list of (H, W, 3) uint8 numpy
    images - another external, out-of-domain "change-source" material/texture
    pool for draem_perlin(), similar in spirit to DTD but larger and more
    diverse (23 material classes, e.g. wood, metal, fabric, brick, glass) -
    see SyntheticCDDataModule's `change_source` option.

    Downloads via HuggingFace datasets on first use, cached under `data_path`.
    """
    from datasets import load_dataset

    ds = load_dataset("mcimpoi/minc-2500_split_1", split="train", cache_dir=str(data_path))
    return [np.array(row["image"].convert("RGB")) for row in ds]
