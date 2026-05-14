"""
PyTorch Dataset for Multimodal IPO Data.

Loads pre-computed text, images, and tabular features for each IPO and
returns them in a format suitable for the multimodal fusion model.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class IPOMultimodalDataset(Dataset):
    """
    Multimodal dataset yielding (images, text, tabular, targets) per IPO.

    Args:
        universe_path: Path to ipo_universe.parquet.
        text_dir: Directory with <cik>/<accession>_sections.json files.
        image_manifest_path: JSON mapping cik → list of image paths.
        image_transform: torchvision transform for image preprocessing.
        text_section: Which S-1 section to use (default: Risk Factors).
        max_images: Maximum images per firm.
        tabular_cols: List of numeric column names for tabular features.
        target_cols: List of target column names.
    """

    def __init__(
        self,
        universe_path: str | Path,
        text_dir: str | Path = "data/processed/text",
        image_manifest_path: str | Path = "data/processed/images/image_manifest.json",
        image_transform=None,
        text_section: str = "Risk Factors",
        max_images: int = 16,
        tabular_cols: list[str] | None = None,
        target_cols: list[str] | None = None,
    ):
        self.df = pd.read_parquet(universe_path)
        self.text_dir = Path(text_dir)
        self.text_section = text_section
        self.max_images = max_images
        self.image_transform = image_transform

        # Load image manifest
        manifest_path = Path(image_manifest_path)
        if manifest_path.exists():
            self.image_manifest = json.loads(manifest_path.read_text())
        else:
            self.image_manifest = {}
            logger.warning("No image manifest at %s", manifest_path)

        # Define feature and target columns
        self.tabular_cols = tabular_cols or [
            "offer_size", "firm_age", "underwriter_rank", "vc_backed",
            "log_assets", "leverage", "rnd_intensity", "revenue_growth",
        ]
        self.target_cols = target_cols or [
            "first_day_return", "broken_ipo", "post_ipo_volatility_6m",
        ]

        # Precompute tabular features (fill NaN with 0, standardize)
        available_tab_cols = [c for c in self.tabular_cols if c in self.df.columns]
        self.tabular_data = self.df[available_tab_cols].fillna(0).values.astype(np.float32)

        # Standardize
        self._tab_mean = self.tabular_data.mean(axis=0)
        self._tab_std = self.tabular_data.std(axis=0) + 1e-8
        self.tabular_data = (self.tabular_data - self._tab_mean) / self._tab_std

        # Targets
        available_target_cols = [c for c in self.target_cols if c in self.df.columns]
        self.targets = self.df[available_target_cols].fillna(0).values.astype(np.float32)

        logger.info(
            "Dataset: %d IPOs | %d tabular features | %d targets | %d firms with images",
            len(self.df),
            self.tabular_data.shape[1],
            self.targets.shape[1] if self.targets.ndim > 1 else 1,
            len(self.image_manifest),
        )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        cik = str(row.get("cik", ""))

        # --- Text ---
        text = self._load_text(cik)

        # --- Images ---
        images, image_mask = self._load_images(cik)

        # --- Tabular ---
        tabular = torch.tensor(self.tabular_data[idx], dtype=torch.float32)

        # --- Targets ---
        targets = torch.tensor(self.targets[idx], dtype=torch.float32)

        return {
            "text": text,
            "images": images,         # (max_images, C, H, W)
            "image_mask": image_mask,  # (max_images,) bool mask of valid images
            "tabular": tabular,        # (n_features,)
            "targets": targets,        # (n_targets,)
            "cik": cik,
            "ticker": str(row.get("ticker", "")),
        }

    def _load_text(self, cik: str) -> str:
        """Load the Risk Factors section text for a CIK."""
        cik_dir = self.text_dir / cik
        if not cik_dir.exists():
            return ""

        # Find the most recent sections JSON
        section_files = sorted(cik_dir.glob("*_sections.json"), reverse=True)
        if not section_files:
            return ""

        sections = json.loads(section_files[0].read_text())
        return sections.get(self.text_section, "")

    def _load_images(self, cik: str) -> tuple[torch.Tensor, torch.Tensor]:
        """Load and preprocess images for a CIK, padding to max_images."""
        image_paths = self.image_manifest.get(cik, [])

        # Default empty tensors
        placeholder = torch.zeros(self.max_images, 3, 224, 224)
        mask = torch.zeros(self.max_images, dtype=torch.bool)

        for i, path in enumerate(image_paths[: self.max_images]):
            try:
                img = Image.open(path).convert("RGB")
                if self.image_transform is not None:
                    img = self.image_transform(img)
                else:
                    # Default CLIP-compatible transform
                    from torchvision import transforms
                    default_tf = transforms.Compose([
                        transforms.Resize((224, 224)),
                        transforms.ToTensor(),
                        transforms.Normalize(
                            [0.48145466, 0.4578275, 0.40821073],
                            [0.26862954, 0.26130258, 0.27577711],
                        ),
                    ])
                    img = default_tf(img)
                placeholder[i] = img
                mask[i] = True
            except Exception as e:
                logger.debug("Failed to load image %s: %s", path, e)
                continue

        return placeholder, mask

    def get_tabular_dim(self) -> int:
        """Return the dimensionality of tabular features."""
        return self.tabular_data.shape[1]

    def get_num_targets(self) -> int:
        """Return the number of target columns."""
        return self.targets.shape[1] if self.targets.ndim > 1 else 1


def build_dataloaders(
    universe_path: str | Path,
    train_end: str = "2018-12-31",
    val_start: str = "2019-01-01",
    val_end: str = "2020-12-31",
    test_start: str = "2021-01-01",
    batch_size: int = 32,
    num_workers: int = 4,
    **dataset_kwargs,
) -> tuple:
    """Build train/val/test DataLoaders with time-based splits."""
    from torch.utils.data import DataLoader

    # Load full universe and split
    full_df = pd.read_parquet(universe_path)

    from src.data.ipo_universe import time_split
    train_df, val_df, test_df = time_split(full_df, train_end, val_start, val_end, test_start)

    # Save split DataFrames temporarily
    for split_name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        tmp_path = Path(universe_path).parent / f"ipo_{split_name}.parquet"
        split_df.to_parquet(tmp_path, index=False)

    datasets = {}
    for split_name in ["train", "val", "test"]:
        tmp_path = Path(universe_path).parent / f"ipo_{split_name}.parquet"
        datasets[split_name] = IPOMultimodalDataset(
            universe_path=tmp_path,
            **dataset_kwargs,
        )

    loaders = {
        "train": DataLoader(
            datasets["train"], batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True, drop_last=False,
        ),
        "val": DataLoader(
            datasets["val"], batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True,
        ),
        "test": DataLoader(
            datasets["test"], batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True,
        ),
    }
    return loaders, datasets
