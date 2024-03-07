import torch
from torch.utils.data import DataLoader, Dataset
from shared import (
    ROOT_DIR, TRAIN, VALIDATION, TEST, DATALOADER, BATCH_SIZE, DEVICE,
    AUGMENTATION_LIST,
    AUGMENTATION_H_ROLL_WRAPPED,
    AUGMENTATION_FLIP
)
from augmentations import augment_wrap_roll, augment_flip
from typing import Tuple, Optional, Union
from pathlib import Path
import numpy as np
IMAGES_FOLDER = "images"
LABELS_FOLDER = "labels"


def load_npy_files(path: Path) -> torch.Tensor:
    """Load .npy file from disk into a tensor

    Args:
        path (Path): path to .npy file

    Returns:
        torch.Tensor: tensor - on CPU
        Note that entire dataset could be loaded in memory
    """
    if path is None:
        return torch.empty(0)
    assert path.exists(), f"Path {path} does not exist"
    return torch.from_numpy(np.load(str(path))).unsqueeze(0).float()


class SegmentationDataset(Dataset):
    def __init__(
        self,
        images_path: Path = ROOT_DIR/"data"/TRAIN/IMAGES_FOLDER,
        labels_path: Optional[Path] = None,
        device: str = DEVICE,
        preloaded: bool = False,
        augmentation_list: Optional[list] = [],
        sanity_check: bool = True
    ):
        self.preloaded = preloaded
        self.augmentation_list = augmentation_list
        self.device = device
        img_list = sorted(list(images_path.glob("*.npy")))
        if labels_path is not None:
            label_list = sorted(list(labels_path.glob("*.npy")))
            label_list = [labels_path/img.name for img in img_list if (labels_path/img.name).exists()]
            assert len(label_list) == len(
                img_list), f"Number of images {len(img_list)} and labels {len(label_list)} do not match"
        else:
            label_list = [None for _ in range(len(img_list))]
        print(f"TOTAL ELEMENTS {len(img_list)}")
        self.path_list = list(zip(img_list, label_list))

        if sanity_check:
            new_list = []
            for img_path, label_path in self.path_list:
                img = load_npy_files(img_path)
                # mini, maxi = torch.min(img), torch.max(img)
                # print(mini, maxi)
                if torch.isnan(img).any():
                    print("Got NaN in image", img_path)
                    continue
                else:
                    new_list.append((img_path, label_path))
            self.path_list = new_list
        self.n_samples = len(self.path_list)
        # If we can preload everything in memory, we can do it
        if preloaded:
            self.data_list = [(load_npy_files(img_path), load_npy_files(label_path))
                              for img_path, label_path in self.path_list]
        else:
            self.data_list = self.path_list

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, Union[torch.Tensor, None]]:
        """Access a specific (image, label mask) pair element of the dataset

        Args:
            index (int): access index

        Returns:
            Tuple[torch.Tensor, Union[torch.Tensor, None]]: image , label
            [1, 36, 36] image tensor
        """
        if self.preloaded:
            img_data, label_data = self.data_list[index]
        else:
            img_data = load_npy_files(self.data_list[index][0])
            label_data = load_npy_files(self.data_list[index][1])
        if AUGMENTATION_H_ROLL_WRAPPED in self.augmentation_list:
            img_data, label_data = augment_wrap_roll(img_data, label_data)
        if AUGMENTATION_FLIP in self.augmentation_list:
            img_data, label_data = augment_flip(img_data, label_data)
        return (
            img_data.to(self.device),
            label_data.to(self.device) if len(label_data) > 0 else self.path_list[index][0].stem
        )

    def __len__(self):
        return self.n_samples


def get_dataloaders(config: dict, device: str = DEVICE):
    augmentation_list = config[DATALOADER].get(AUGMENTATION_LIST, [])
    if len(augmentation_list) > 0:
        print(f"Using augmentations {augmentation_list}")
    dl_train = SegmentationDataset(
        ROOT_DIR/"data"/TRAIN/IMAGES_FOLDER,
        labels_path=ROOT_DIR/"data"/TRAIN/LABELS_FOLDER,
        augmentation_list=augmentation_list,
        device=device
    )
    dl_valid = SegmentationDataset(
        ROOT_DIR/"data"/VALIDATION/IMAGES_FOLDER,
        labels_path=ROOT_DIR/"data"/VALIDATION/LABELS_FOLDER,
        device=device
    )
    dl_test = SegmentationDataset(
        ROOT_DIR/"data"/TEST/IMAGES_FOLDER,
        device=device
    )
    dl_dict = {
        TRAIN: DataLoader(
            dl_train,
            shuffle=True,
            batch_size=config[DATALOADER][BATCH_SIZE][TRAIN],
        ),
        VALIDATION: DataLoader(
            dl_valid,
            shuffle=False,
            batch_size=config[DATALOADER][BATCH_SIZE][VALIDATION]
        ),
        TEST: DataLoader(dl_test, shuffle=False, batch_size=config[DATALOADER][BATCH_SIZE][TEST])
    }
    return dl_dict


if __name__ == "__main__":
    config = {
        DATALOADER: {
            BATCH_SIZE: {
                TRAIN: 4,
                VALIDATION: 4,
                # TEST: 8
            }
        }
    }
    import matplotlib.pyplot as plt
    dl_dict = get_dataloaders(config)
    for run_index in range(2):
        for idx, mode in enumerate([TRAIN, VALIDATION]):
            img, lab = next(iter(dl_dict[mode]))
            plt.subplot(2, 4, run_index*4+1+2*idx)
            plt.imshow(img.view(-1, img.shape[-1]).cpu().numpy(), cmap="gray")
            plt.title(mode + f" input - run {run_index}")
            plt.subplot(2, 4, run_index*4+1+2*idx+1)
            plt.imshow(lab.view(-1, img.shape[-1]).cpu().numpy())
            plt.title(mode + f" label - run {run_index}")
    plt.show()
    # for batch_idx, data in enumerate(dl_dict[VALIDATION]):
    #     print(batch_idx, data[0].shape)
