"""
The file contains all the Dataset instances needed to create the CLAP training set.

Authors:
    - Francesco Paissan 2023, 2024
"""

import csv
import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
import yaml
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.datasets.utils import (download_and_extract_archive,
                                        download_url)
from tqdm import tqdm

duration = 5


def random_crop(tmp, sample_rate):
    crop_len = sample_rate * duration
    if tmp.shape[1] - crop_len > 0:
        start = torch.randint(low=0, high=(tmp.shape[1] - crop_len), size=(1,))
        tmp = tmp[:, start : start + crop_len]

    return tmp


class VGGSoundDataset(Dataset):
    def __init__(
        self, data_folder: Path, sample_rate: int = 44100, split: str = "train"
    ):
        super().__init__()
        folders: List[Path] = list(sorted(data_folder.joinpath(split).iterdir()))

        self.audio: List[Path] = [
            audio_fn for fold in folders for audio_fn in sorted(fold.iterdir())
        ]
        self.name2labels: Dict[str, int] = {
            str(n.name): l for (l, n) in enumerate(folders)
        }
        self.labels2name: Dict[int, str] = {
            l: str(n.name) for (l, n) in enumerate(folders)
        }

        self.sample_rate = sample_rate

    def __len__(self):
        return len(self.audio)

    def __getitem__(self, idx: int):
        tmp, sr = torchaudio.load(self.audio[idx])
        resample = T.Resample(sr, self.sample_rate)
        tmp = resample(tmp)

        zeros = (duration * self.sample_rate) - tmp.shape[1]
        tmp = F.pad(tmp, (0, zeros))

        return (
            tmp,
            "this is the sound of " + str(self.audio[idx].parent).split("/")[-1],
        )


class AudioCapsDataset(Dataset):
    def __init__(
        self, data_folder: Path, sample_rate: int = 44100, split: str = "train"
    ):
        super().__init__()
        self.base_folder: Path = data_folder.joinpath(split)
        self.meta = pd.read_csv(data_folder.joinpath(f"{split}.csv"))[
            ["youtube_id", "caption"]
        ]
        actually_downloaded = [
            os.path.exists(self.base_folder.joinpath(f"{y}.wav"))
            for y in self.meta.youtube_id
        ]
        print(
            "INFO: %d samples missing in AudioCaps."
            % (len(self.meta) - sum(actually_downloaded))
        )
        self.meta = self.meta.iloc[actually_downloaded]
        self.sample_rate = sample_rate

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx: int):
        yt_id = self.meta.iloc[idx].youtube_id
        caption = self.meta.iloc[idx].caption

        fp = self.base_folder.joinpath(f"{yt_id}.wav")
        tmp, sr = torchaudio.load(fp)
        resample = T.Resample(sr, self.sample_rate)
        tmp = resample(tmp)

        tmp = random_crop(tmp, self.sample_rate)
        zeros = (duration * self.sample_rate) - tmp.shape[1]
        tmp = F.pad(tmp, (0, zeros))
        tmp = tmp.sum(0, keepdims=True)  # make sure all clips have 1 channel

        return tmp, caption


class ClothoV2(Dataset):
    def __init__(
        self,
        data_folder: Path,
        sample_rate: int = 44100,
        metadata_path: str = "clotho_captions_development.csv",
    ):
        super().__init__()
        self.base_folder: Path = data_folder.joinpath("data")
        self.sample_rate = sample_rate
        self.filenames = []
        self.captions = []

        # Open the CSV file and read its contents
        with open(data_folder.joinpath(metadata_path), newline="") as csvfile:
            csvreader = csv.reader(csvfile)

            # Skip the header row if there is one
            next(csvreader, None)

            # Iterate over each row in the CSV file
            for row in csvreader:
                # Append each column's data to its corresponding list
                filename, *captions = row
                assert len(captions) == 5

                self.filenames.extend([filename] * len(captions))
                self.captions.extend(captions)

        assert len(self.filenames) == len(self.captions)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx: int):
        fp = self.base_folder.joinpath(self.filenames[idx])
        tmp, sr = torchaudio.load(fp)
        resample = T.Resample(sr, self.sample_rate)
        tmp = resample(tmp)

        tmp = random_crop(tmp, self.sample_rate)

        zeros = (duration * self.sample_rate) - tmp.shape[1]
        tmp = F.pad(tmp, (0, zeros))
        tmp = tmp.sum(0, keepdims=True)  # make sure all clips have 1 channel

        return tmp, self.captions[idx]


class MACs(Dataset):
    def __init__(
        self,
        data_folder: Path,
        sample_rate: int = 44100,
        metadata_path: str = "MACS.yaml",
    ):
        super().__init__()
        self.base_folder: Path = data_folder.joinpath("data")
        dat = yaml.load(
            open(self.base_folder.parent.joinpath(metadata_path)),
            Loader=yaml.SafeLoader,
        )
        filenames = []
        captions = []
        for f in dat["files"]:
            for c in f["annotations"]:
                filenames.append(f["filename"])
                captions.append(c["sentence"])
        self.captions = captions
        self.filenames = filenames
        self.sample_rate = sample_rate

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx: int):
        fp = self.base_folder.joinpath(self.filenames[idx])
        tmp, sr = torchaudio.load(fp)
        resample = T.Resample(sr, self.sample_rate)
        tmp = resample(tmp)

        tmp = random_crop(tmp, self.sample_rate)

        zeros = (duration * self.sample_rate) - tmp.shape[1]
        tmp = F.pad(tmp, (0, zeros))
        tmp = tmp.sum(0, keepdims=True)  # make sure all clips have 1 channel

        return tmp, self.captions[idx]


class FSD50k(Dataset):
    def __init__(
        self,
        data_folder: Path,
        sample_rate: int = 44100,
        metadata_path: str = "dev_clips_info_FSD50K.json",
    ):
        super().__init__()
        self.base_folder: Path = data_folder.joinpath("FSD50K.dev_audio")
        dat = json.load(open(self.base_folder.parent.joinpath(metadata_path)))
        filenames = []
        captions = []
        for f in dat.keys():
            filenames.append(f + ".wav")
            captions.append(
                (dat[f]["title"] + " " + dat[f]["description"])
                .replace("wav", "")
                .replace("WAV", "")
            )
        self.captions = captions
        self.filenames = filenames
        self.sample_rate = sample_rate

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx: int):
        fp = self.base_folder.joinpath(self.filenames[idx])
        tmp, sr = torchaudio.load(fp)
        resample = T.Resample(sr, self.sample_rate)
        tmp = resample(tmp)

        tmp = random_crop(tmp, self.sample_rate)

        zeros = (duration * self.sample_rate) - tmp.shape[1]
        tmp = F.pad(tmp, (0, zeros))
        tmp = tmp.sum(0, keepdims=True)  # make sure all clips have 1 channel

        return tmp, self.captions[idx]


class TUT17(Dataset):
    def __init__(
        self,
        data_folder: Path,
        sample_rate: int = 44100,
        metadata_path: str = "evaluate.txt",
    ):
        super().__init__()
        self.sample_rate = sample_rate

        if not isinstance(data_folder, Path):
            data_folder = Path(data_folder)

        self.base_folder: Path = data_folder.joinpath("data")
        self.dat = pd.read_csv(
            self.base_folder.parent.joinpath(metadata_path), sep="\t"
        )
        self.classes = [x.replace("_", " ") for x in list(self.dat["class"].unique())]

    def __len__(self):
        return len(self.dat)

    def __getitem__(self, idx: int):
        fp = self.base_folder.joinpath(self.dat["filename"].iloc[idx])

        tmp, sr = torchaudio.load(fp)
        resample = T.Resample(sr, self.sample_rate)
        tmp = resample(tmp)

        tmp = random_crop(tmp, self.sample_rate)

        zeros = (duration * self.sample_rate) - tmp.shape[1]
        tmp = F.pad(tmp, (0, zeros))
        tmp = tmp.sum(0, keepdims=True)  # make sure all clips have 1 channel

        class_id = torch.tensor(
            self.classes.index(self.dat["class"].iloc[idx].replace("_", " "))
        )

        uttid = str(fp.name).split(".")[0]

        return (
            tmp,
            f"this is the sound of {self.classes[class_id]}",
            class_id,
            uttid,
        )


# thanks https://github.com/multitel-ai/urban-sound-classification-and-comparison
class UrbanSound8K(Dataset):
    base_folder = "UrbanSound8K"
    resources = [
        (
            "https://zenodo.org/record/1203745/files/UrbanSound8K.tar.gz",
            "9aa69802bbf37fb986f71ec1483a196e",
        )
    ]

    def __init__(
        self,
        dataset_folder,
        sample_rate=44100,
        fold=None,
        transform=None,
        download=False,
    ):
        super().__init__()
        self.dataset_folder = dataset_folder
        self.sample_rate = sample_rate
        # os.path.join(dataset_folder, UrbanSound8K.base_folder)

        self.path_to_csv = os.path.join(
            self.dataset_folder, "metadata/UrbanSound8K.csv"
        )
        self.path_to_audio_folder = os.path.join(self.dataset_folder, "audio")
        self.path_to_melTALNet = os.path.join(self.dataset_folder, "melTALNet")

        # Downloading and extracting if needed
        if download:
            self.download()

        # Checking if the dataset exist at specified location
        if not os.path.exists(self.dataset_folder):
            raise RuntimeError(
                "Dataset not found." + " You can use download=True to download it"
            )

        self.raw_annotations = pd.read_csv(self.path_to_csv)
        self.fold = fold

        self.transform = transform
        self.classes = [
            "air_conditioner",
            "car_horn",
            "children_playing",
            "dog_bark",
            "drilling",
            "engine_idling",
            "gun_shot",
            "jackhammer",
            "siren",
            "street_music",
        ]

    def train_validation_split(self):
        """ """
        train_idx = list(
            self.raw_annotations[self.raw_annotations["fold"] != self.fold].index
        )
        val_idx = list(
            self.raw_annotations[self.raw_annotations["fold"] == self.fold].index
        )

        train_set = Subset(self, train_idx)
        val_set = Subset(self, val_idx)
        val_set.transform = None

        return train_set, val_set

    def __len__(self):
        return len(self.raw_annotations)

    def download(self):
        if os.path.exists(os.path.join(self.dataset_folder, "metadata")):
            return

        # Download files
        for url, md5 in self.resources:
            down_root = os.path.dirname(self.dataset_folder)
            download_and_extract_archive(
                url,
                download_root=down_root,
                filename=self.base_folder + ".tar.gz",
                md5=md5,
                remove_finished=True,
            )

    def __getitem__(self, index):
        file_name = self.raw_annotations["slice_file_name"].iloc[index]
        file_path = os.path.join(
            os.path.join(
                self.path_to_audio_folder,
                "fold" + str(self.raw_annotations["fold"].iloc[index]),
            ),
            file_name,
        )

        wav, sr = torchaudio.load(file_path)
        resample = T.Resample(sr, self.sample_rate)

        tmp = resample(wav)

        tmp = random_crop(tmp, self.sample_rate)

        zeros = (duration * self.sample_rate) - tmp.shape[1]
        tmp = F.pad(tmp, (0, zeros))
        tmp = tmp.sum(0, keepdims=True)  # make sure all clips have 1 channel

        idx = torch.tensor(self.raw_annotations["classID"].iloc[index])

        uttid = file_name.split(".")[0]

        return (
            tmp,
            f"this is the sound of {self.classes[idx]}",
            torch.tensor(self.raw_annotations["classID"].iloc[index]),
            uttid,
        )


class ESC50(Dataset):
    base_folder = ""
    url = "https://github.com/karolpiczak/ESC-50/archive/refs/heads/master.zip"
    filename = "ESC-50-master.zip"
    audio_dir = "audio"
    label_col = "category"
    file_col = "filename"
    meta = {
        "filename": os.path.join("meta", "esc50.csv"),
    }

    def __init__(
        self,
        root,
        sample_rate: int = 44100,
        reading_transformations: Optional[nn.Module] = None,
        download: bool = False,
        train: bool = True,
        cut_off: int = 500,
    ):
        super().__init__()
        self.root = os.path.expanduser(root)
        if download:
            self.download()

        self._load_meta()

        self.targets, self.audio_paths = [], []
        self.pre_transformations = reading_transformations
        print("Loading audio files")

        self.df["category"] = self.df["category"].str.replace("_", " ")

        for _, row in tqdm(self.df.iterrows()):
            file_path = os.path.join(
                self.root, self.base_folder, self.audio_dir, row[self.file_col]
            )
            self.targets.append(row[self.label_col])
            self.audio_paths.append(file_path)

        self.sample_rate = sample_rate

    def _load_meta(self):
        path = os.path.join(self.root, self.base_folder, self.meta["filename"])

        self.df = pd.read_csv(path)
        self.class_to_idx = {}
        self.classes = [
            x.replace("_", " ") for x in sorted(self.df[self.label_col].unique())
        ]
        for i, category in enumerate(self.classes):

            self.class_to_idx[category] = i

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        file_path, target = self.audio_paths[index], self.targets[index]
        idx = torch.tensor(self.class_to_idx[target])
        tmp, sr = torchaudio.load(file_path)
        resample = T.Resample(sr, self.sample_rate)
        tmp = resample(tmp)

        tmp = random_crop(tmp, self.sample_rate)

        zeros = (duration * self.sample_rate) - tmp.shape[1]
        tmp = F.pad(tmp, (0, zeros))
        tmp = tmp.sum(0, keepdims=True)  # make sure all clips have 1 channel

        utt_id = self.audio_paths[index].split("/")[-1].split(".")[0]

        return tmp, "this is the sound of " + target, idx, utt_id

    def __len__(self):
        return len(self.audio_paths)

    def download(self):
        download_url(self.url, self.root, self.filename)

        # extract file
        from zipfile import ZipFile

        with ZipFile(os.path.join(self.root, self.filename), "r") as zip:
            zip.extractall(path=self.root)


def prepare_clap_datasets(hparams):
    dataset_list = []
    if hparams["audiocaps_folder"] is not None:
        audiocaps = AudioCapsDataset(
            Path(hparams["audiocaps_folder"]).joinpath("data"),
            sample_rate=hparams["sample_rate"],
        )
        dataset_list.append(audiocaps)

    if hparams["clotho_folder"] is not None:
        clotho = ClothoV2(
            Path(hparams["clotho_folder"]), sample_rate=hparams["sample_rate"]
        )
        dataset_list.append(clotho)

    if hparams["macs_folder"] is not None:
        macs = MACs(Path(hparams["macs_folder"]), sample_rate=hparams["sample_rate"])
        dataset_list.append(macs)

    if hparams["fsd50k_folder"] is not None:
        fsd = FSD50k(Path(hparams["fsd50k_folder"]), sample_rate=hparams["sample_rate"])
        dataset_list.append(fsd)

    train_data = None
    if len(dataset_list) != 0:
        train_data = torch.utils.data.ConcatDataset(dataset_list)

    if hparams["zs_eval"]:
        assert not (
            hparams["esc_folder"] is None
            and hparams["us8k_folder"] is None
            and hparams["tut17_folder"] is None
        ), "Select one ZS eval dataset."

        if (
            hparams["esc_folder"] is not None
            and hparams["us8k_folder"] is not None
            and hparams["tut17_folder"] is not None
        ):
            print("You should select only one benchmark!")
            quit()

        if hparams["esc_folder"] is not None:
            print("INFO: Running eval on the ESC-50 benchmark.")
            train_data = ESC50(
                hparams["esc_folder"], sample_rate=hparams["sample_rate"]
            )
        if hparams["us8k_folder"] is not None:
            print("INFO: Running eval on the UrbanSound8K benchmark.")
            train_data = UrbanSound8K(
                hparams["us8k_folder"], sample_rate=hparams["sample_rate"]
            )

        if hparams["tut17_folder"] is not None:
            print("INFO: Running eval on the UrbanSound8K benchmark.")
            train_data = TUT17(
                hparams["tut17_folder"], sample_rate=hparams["sample_rate"]
            )

        return {
            "train": DataLoader(
                train_data,
                batch_size=hparams["batch_size"],
                pin_memory=True,
                persistent_workers=True,
                num_workers=128,
            ),
        }

    if train_data is None:
        print("Data config is not valid.")

    return {
        "train": DataLoader(
            train_data,
            batch_size=hparams["batch_size"],
            drop_last=True,
            shuffle=True,
            pin_memory=True,
            persistent_workers=True,
            num_workers=8,
        ),
    }
