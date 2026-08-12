from pathlib import Path

import cv2
import lightning as L
from torch.utils.data import DataLoader

from data.dataset import CDDataset
from data.synthetic_change import (
    SyntheticChangeDataset,
    DATASET_CALIBRATION,
    DEFAULT_CALIBRATION,
)
from data.transforms import build_transforms

from data.hf_datasets import (
    get_levircd,
    get_sysu,
    get_egybcd,
    get_clcd,
    get_gvlm,
    get_oscd96,
)


def _worker_init_fn(worker_id):
    # Prevent OpenCV's internal thread pool from fighting with DataLoader
    # worker processes; on Windows this combination corrupts worker heaps
    # and causes spurious tiny-array allocation failures.
    cv2.setNumThreads(0)


class CDDataModule(L.LightningDataModule):
    def __init__(
        self,
        config,
        pretrain: bool,
        data_path: str = None,
        use_hf=True,
        load_in_mem=False,
    ):
        super().__init__()

        self.config = config
        self.data_path = Path(data_path)
        self.pretrain = pretrain
        self.use_hf = use_hf
        self.load_in_mem = load_in_mem

        self.batch_size = config.data.batch_size

        self.prepare_dataset()

        print(f"Num workers: {config.data.num_workers}")

    def prepare_dataset(self):
        if self.pretrain:
            self.dataset_name = self.config.data.pretrain_dataset
        else:
            self.dataset_name = self.config.data.dataset

        if self.use_hf:
            data = get_hf_dataset(self.dataset_name, self.data_path)
            data.set_format("numpy")
        else:
            data = {
                "train": self.data_path / self.dataset_name / "train",
                "test": self.data_path / self.dataset_name / "test",
                "val": self.data_path / self.dataset_name / "val",
            }

        train_transforms = build_transforms(
            self.config, pretrain=self.pretrain, test=False
        )
        test_transforms = build_transforms(
            self.config, pretrain=self.pretrain, test=True
        )

        self.train_data = CDDataset(
            data["train"],
            train_transforms,
            use_hf=self.use_hf,
            load_in_mem=self.load_in_mem,
        )
        self.test_data = CDDataset(
            data["test"],
            test_transforms,
            use_hf=self.use_hf,
            load_in_mem=self.load_in_mem,
        )

        if (
            (self.use_hf and "val" in data)
            or (self.load_in_mem == "direct" and data["val"].exists())
            or (self.load_in_mem == "hdf5" and Path(str(data["val"]) + ".h5").exists())
        ):
            self.val_data = CDDataset(
                data["val"],
                test_transforms,
                use_hf=self.use_hf,
                load_in_mem=self.load_in_mem,
            )
        else:
            print("Validation data not present, using test set.")
            self.val_data = self.test_data

    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            num_workers=self.config.data.num_workers,
            # pin_memory=self.config.data.pin_memory,
            drop_last=True,
            shuffle=True,
            persistent_workers=self.config.data.num_workers > 0,
            worker_init_fn=_worker_init_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_data,
            batch_size=self.batch_size,
            num_workers=self.config.data.num_workers,
            # pin_memory=self.config.data.pin_memory,
            persistent_workers=self.config.data.num_workers > 0,
            worker_init_fn=_worker_init_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_data,
            batch_size=self.batch_size,
            num_workers=self.config.data.num_workers,
            # pin_memory=self.config.data.pin_memory,
            persistent_workers=self.config.data.num_workers > 0,
            worker_init_fn=_worker_init_fn,
        )


class SyntheticCDDataModule(CDDataModule):
    """
    Self-supervised variant of CDDataModule: train/val are synthetic
    (imageA, imageB, label) triples generated on the fly via cutpaste/cutmix
    from the target dataset's own images (no real change labels used), while
    test stays the real dataset - so the normal train.py eval step at the end
    of a run already produces the real zero-shot metric.

    Val uses a fixed `val_seed` so the same synthetic pairs are regenerated
    each epoch (a stable signal to track); train regenerates fresh every time.
    """

    def __init__(
        self,
        config,
        data_path: str = None,
        use_hf=True,
        load_in_mem=False,
        synth_method: str = "cutpaste",
        change_source: str = "self",
        soft_edges: bool = False,
        color_match: bool = False,
        val_seed: int = 42,
    ):
        self.synth_method = synth_method
        self.change_source = change_source
        self.soft_edges = soft_edges
        self.color_match = color_match
        self.val_seed = val_seed
        super().__init__(
            config,
            pretrain=False,
            data_path=data_path,
            use_hf=use_hf,
            load_in_mem=load_in_mem,
        )

    def prepare_dataset(self):
        self.dataset_name = self.config.data.dataset

        data = get_hf_dataset(self.dataset_name, self.data_path)
        data.set_format("numpy")

        # pool both timepoints as target images - both are just single
        # unlabeled satellite images from the encoder's point of view
        target_images = list(data["train"]["imageA"]) + list(data["train"]["imageB"])

        # "self" (default): reuse the target dataset's own images as the
        # cutmix/draem donor content too (same as before). "dtd": use an
        # external, out-of-domain texture pool instead - see data/dtd_source.py.
        # "nwpu": use NWPU-RESISC45, an external but IN-DOMAIN (real aerial/
        # satellite scenes) pool instead - see data/nwpu_source.py.
        if self.change_source == "dtd":
            from data.dtd_source import get_dtd_images

            donor_images = get_dtd_images(self.data_path)
            print(f"Using DTD as change-source pool: {len(donor_images)} textures")
        elif self.change_source == "nwpu":
            from data.nwpu_source import get_nwpu_images

            donor_images = get_nwpu_images(self.data_path)
            print(f"Using NWPU-RESISC45 as change-source pool: {len(donor_images)} scenes")
        elif self.change_source == "ucmerced":
            from data.ucmerced_source import get_ucmerced_images

            donor_images = get_ucmerced_images(self.data_path)
            print(f"Using UC Merced Land Use as change-source pool: {len(donor_images)} scenes")
        elif self.change_source == "self":
            donor_images = None  # SyntheticChangeDataset falls back to target_images
        else:
            raise ValueError(
                f"Unknown change_source {self.change_source}, expected 'self', 'dtd', 'nwpu' or 'ucmerced'"
            )

        # dataset-specific calibration of change area/frequency, measured from
        # each dataset's real (train-split) label masks - see synthetic_change.py
        calib = DATASET_CALIBRATION.get(self.dataset_name, DEFAULT_CALIBRATION)
        print(f"Synthetic change calibration for {self.dataset_name}: {calib}")

        train_transforms = build_transforms(self.config, pretrain=False, test=False)
        test_transforms = build_transforms(self.config, pretrain=False, test=True)

        self.train_data = SyntheticChangeDataset(
            target_images,
            train_transforms,
            method=self.synth_method,
            soft_edges=self.soft_edges,
            color_match=self.color_match,
            source_images=donor_images,
            **calib,
        )
        self.val_data = SyntheticChangeDataset(
            target_images,
            test_transforms,
            method=self.synth_method,
            soft_edges=self.soft_edges,
            color_match=self.color_match,
            source_images=donor_images,
            **calib,
            seed=self.val_seed,
        )
        self.test_data = CDDataset(
            data["test"],
            test_transforms,
            use_hf=self.use_hf,
            load_in_mem=self.load_in_mem,
        )


def get_hf_dataset(name, data_path):
    if name == "oscd96":
        ds = get_oscd96(data_path)
    elif name == "levir":
        ds = get_levircd(data_path)
    elif name == "sysu":
        ds = get_sysu(data_path)
    elif name == "egybcd":
        ds = get_egybcd(data_path)
    elif name == "clcd":
        ds = get_clcd(data_path)
    elif name == "gvlm":
        ds = get_gvlm(data_path)
    else:
        msg = (
            f"Unknown HuggingFace dataset {name}. If you want to use directory or hdf5 dataset, set `data.use_hf` to False."
            f"Optionally set `data.load_in_mem` to 'direct' or 'hdf5'"
        )
        raise ValueError(msg)

    for key in ds:
        ds[key] = ds[key].add_column("img_idx", range(len(ds[key])))

    return ds
