from dataclasses import dataclass

import numpy as np
import torch
import torchaudio
from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import os
torch.set_num_threads(1)


@dataclass
class DataConfig:
    filelist_path: str
    sampling_rate: int
    num_samples: int
    batch_size: int
    num_workers: int
    max_len: int


class VocosDataModule(LightningDataModule):
    def __init__(self, train_params: DataConfig, val_params: DataConfig):
        super().__init__()
        self.train_config = train_params
        self.val_config = val_params

    def _get_dataloder(self, cfg: DataConfig, train: bool):
        dataset = VocosDataset(cfg, train=train)
        dataloader = DataLoader(
            dataset, batch_size=cfg.batch_size, num_workers=cfg.num_workers, shuffle=train, pin_memory=True,
        )
        return dataloader

    def train_dataloader(self) -> DataLoader:
        return self._get_dataloder(self.train_config, train=True)

    def val_dataloader(self) -> DataLoader:
        return self._get_dataloder(self.val_config, train=False)


class VocosDataset(Dataset):
    def __init__(self, cfg: DataConfig, train: bool):
        with open(cfg.filelist_path) as f:
            self.filelist = f.read().splitlines()
        self.sampling_rate = cfg.sampling_rate
        self.num_samples = cfg.num_samples
        self.max_len = cfg.max_len
        self.train = train
        self.wavs = []
        self.read_data()

    def read_data(self):
        for wav_p in tqdm(self.filelist, desc=f"train={self.train} file loaded: ",mininterval=2):
            y, sr = torchaudio.load(wav_p)
            if y.size(0) > 1:
                # mix to mono
                y = y.mean(dim=0, keepdim=True)       
            if sr != self.sampling_rate:
                y = torchaudio.functional.resample(y, orig_freq=sr, new_freq=self.sampling_rate)     
            if y.size(-1) < self.num_samples:
                pad_length = self.num_samples - y.size(-1)
                padding_tensor = y.repeat(1, 1 + pad_length // y.size(-1))
                y = torch.cat((y, padding_tensor[:, :pad_length]), dim=1)
            self.wavs.append(y)
        np.random.shuffle(self.wavs)
            
    def __len__(self) -> int:
        return min(len(self.filelist), self.max_len)

    def __getitem__(self, index: int) -> torch.Tensor:
        y = self.wavs[index]
        if self.train:
            sr = self.sampling_rate
            gain = np.random.uniform(-1, -6) if self.train else -3
            y, _ = torchaudio.sox_effects.apply_effects_tensor(y, sr, [["norm", f"{gain:.2f}"]])
            start = np.random.randint(low=0, high=y.size(-1) - self.num_samples + 1)
            y = y[:, start : start + self.num_samples]
            if index == 0:
                np.random.shuffle(self.wavs)
        else:
            # During validation, take always the first segment for determinism
            y = y[:, : self.num_samples]

        return y[0]
