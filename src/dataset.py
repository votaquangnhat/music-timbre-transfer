from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset

import audio_wrapper as aw


class TimbreSpectrogramDataset(Dataset):
    def __init__(
        self,
        spectrograms_dir: str,
        timbre: Optional[str] = None,
        subset: Optional[str] = None,
        image_size: int = 512,
        max_samples: Optional[int] = None,
        seed: int = 42,
        verify_files: bool = False,
    ):
        self.spectrograms_dir = Path(spectrograms_dir)
        self.timbre = timbre
        self.subset = subset
        self.image_size = image_size
        self.converter = aw.SpectrogramConverter(image_width=image_size, n_mels=image_size)

        metadata_path = self.spectrograms_dir / "spectrogram_metadata.csv"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

        self.metadata = pd.read_csv(metadata_path)
        self._check_required_columns()

        if timbre is not None:
            self.metadata = self.metadata[self.metadata["timbres"] == timbre]

        if subset is not None:
            self.metadata = self.metadata[self.metadata["subset"] == subset]

        if len(self.metadata) == 0:
            raise ValueError(
                "No spectrogram samples found after filtering. "
                f"timbre={timbre!r}, subset={subset!r}, dir={str(self.spectrograms_dir)!r}"
            )

        # Deterministic subsampling is useful for fast overfit tests / Colab smoke tests.
        if max_samples is not None and max_samples > 0 and len(self.metadata) > max_samples:
            self.metadata = self.metadata.sample(n=max_samples, random_state=seed)

        self.metadata = self.metadata.reset_index(drop=True)

        if verify_files:
            self._verify_files_exist()

    def _check_required_columns(self) -> None:
        required = {"file_name", "timbres", "subset"}
        missing = required - set(self.metadata.columns)
        if missing:
            raise ValueError(
                f"Metadata is missing required columns: {sorted(missing)}. "
                f"Found columns: {list(self.metadata.columns)}"
            )

    def _verify_files_exist(self) -> None:
        missing_files = []
        for file_name in self.metadata["file_name"].tolist():
            path = self.spectrograms_dir / file_name
            if not path.exists():
                missing_files.append(str(path))
                if len(missing_files) >= 10:
                    break
        if missing_files:
            preview = "\n".join(missing_files)
            raise FileNotFoundError(f"Missing spectrogram .npy files, first few:\n{preview}")

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, idx: int) -> dict:
        row = self.metadata.iloc[idx]
        spectrogram_path = self.spectrograms_dir / row["file_name"]
        spectrogram_array = np.load(spectrogram_path)

        image = self.converter.image_from_spectrogram(spectrogram_array)
        if image.size != (self.image_size, self.image_size):
            # Your generation code should already output 512x512. This is only a safety net.
            image = image.resize((self.image_size, self.image_size), Image.Resampling.BICUBIC)

        array = np.asarray(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
        tensor = tensor * 2.0 - 1.0

        return {
            "image": image,
            "tensor": tensor,
            "spectrogram": spectrogram_array,
            "timbre": row["timbres"],
            "filename": row["file_name"],
        }


def collate_fn(batch: list[dict]) -> dict:
    pixel_values = torch.stack([item["tensor"] for item in batch])
    timbres = [item["timbre"] for item in batch]
    filenames = [item["filename"] for item in batch]

    return {
        "pixel_values": pixel_values,
        "timbres": timbres,
        "filenames": filenames,
    }


if __name__ == "__main__":
    dataset = TimbreSpectrogramDataset(
        spectrograms_dir="/home/vtqn/projects/project2/data/spectrogram",
        timbre="piano",
        subset="train",
        verify_files=True,
    )
    print(f"Dataset length: {len(dataset)}")
    sample = dataset[0]
    print(f"Sample keys: {list(sample.keys())}")
    print(f"Sample image size: {sample['image'].size}")
    print(f"Sample spectrogram shape: {sample['spectrogram'].shape}")
    print(f"Sample timbre: {sample['timbre']}")
    print(f"Sample filename: {sample['filename']}")
    print(f"Sample tensor shape: {sample['tensor'].shape}")
