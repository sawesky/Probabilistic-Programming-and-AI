import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, functional


class CVAECIFAR10(Dataset):
    def __init__(self, root, train=True, transform=None, download=False):
        self.original = CIFAR10(root, train=train, download=download)
        self.transform = transform

    def __len__(self):
        return len(self.original)

    def __getitem__(self, item):
        image, label = self.original[item]
        sample = {"original": image, "label": label}
        if self.transform:
            sample = self.transform(sample)

        return sample


class ToTensor:
    def __call__(self, sample):
        sample["original"] = functional.to_tensor(sample["original"])
        sample["label"] = torch.as_tensor(
            np.asarray(sample["label"]), dtype=torch.int64
        )
        return sample


class MaskImages:
    """This transformation masks parts of the CIFAR-10 images for the tutorial."""

    def __init__(self, num_quadrant_inputs, mask_with=-1):
        if num_quadrant_inputs <= 0 or num_quadrant_inputs >= 4:
            raise ValueError("Number of quadrants as inputs must be 1, 2 or 3")
        self.num = num_quadrant_inputs
        self.mask_with = mask_with

    def __call__(self, sample):
        tensor = sample["original"]
        out = tensor.clone()
        _, h, w = tensor.shape

        # Remove the bottom left quadrant from the target output
        out[:, h // 2 :, : w // 2] = self.mask_with
        # If number of quadrants to be used as input is 2, remove the top left quadrant
        if self.num == 2:
            out[:, :, : w // 2] = self.mask_with
        # If number of quadrants to be used as input is 3, remove the top right quadrant
        if self.num == 3:
            out[:, : h // 2, :] = self.mask_with

        # Set the input as complementary
        inp = tensor.clone()
        inp[out != -1] = self.mask_with

        sample["input"] = inp
        sample["output"] = out
        return sample


def get_data_CIFAR10(num_quadrant_inputs, batch_size):
    transforms = Compose(
        [ToTensor(), MaskImages(num_quadrant_inputs=num_quadrant_inputs)]
    )
    datasets, dataloaders, dataset_sizes = {}, {}, {}
    for mode in ["train", "val"]:
        datasets[mode] = CVAECIFAR10(
            "../data", download=True, transform=transforms, train=mode == "train"
        )
        dataloaders[mode] = DataLoader(
            datasets[mode],
            batch_size=batch_size,
            shuffle=mode == "train",
            num_workers=0,
        )
        dataset_sizes[mode] = len(datasets[mode])

    return datasets, dataloaders, dataset_sizes
