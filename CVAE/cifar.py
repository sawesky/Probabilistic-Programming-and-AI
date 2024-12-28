

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, functional
from mnist import MaskImages
import pickle

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def import_cifar10(train=True):
    if train:
      train_data = []
      train_labels = []

      for i in range(1,6):
          data_dict = unpickle(f'./data/cifar-10-batches-py/data_batch_{i}')
          if i == 1:
              train_data = data_dict[b'data']
              train_labels = data_dict[b'labels']
          else:
              train_data = np.vstack((train_data, data_dict[b'data']))
              train_labels = np.hstack((train_labels, data_dict[b'labels']))

      return train_data, train_labels
    else:
      test_data_dict = unpickle(f'./data/cifar-10-batches-py/test_batch')
      test_data = test_data_dict[b'data']
      test_labels = test_data_dict[b'labels']
      return test_data, test_labels

class CIFAR10(Dataset):
    def __init__(self, train=None):
        self.data, self.labels = import_cifar10(train)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        print(self.data[item].shape)
        print(self.data)
        image = self.data[item]
        label = self.labels[item]
        sample = {"original": image, "digit": label}

        return sample
      
class ToTensor:
    def __call__(self, sample):
        sample["original"] = functional.to_tensor(sample["original"])
        sample["digit"] = torch.as_tensor(
            np.asarray(sample["digit"]), dtype=torch.int64
        )
        return sample


class MaskImages:
    """This torchvision image transformation prepares the MNIST digits to be
    used in the tutorial. Depending on the number of quadrants to be used as
    inputs (1, 2, or 3), the transformation masks the remaining (3, 2, 1)
    quadrant(s) setting their pixels with -1. Additionally, the transformation
    adds the target output in the sample dict as the complementary of the input
    """

    def __init__(self, num_quadrant_inputs, mask_with=-1):
        if num_quadrant_inputs <= 0 or num_quadrant_inputs >= 4:
            raise ValueError("Number of quadrants as inputs must be 1, 2 or 3")
        self.num = num_quadrant_inputs
        self.mask_with = mask_with

    def __call__(self, sample):
        tensor = sample["original"].squeeze()
        out = tensor.detach().clone()
        h, w = tensor.shape

        # removes the bottom left quadrant from the target output
        out[h // 2 :, : w // 2] = self.mask_with
        # if num of quadrants to be used as input is 2,
        # also removes the top left quadrant from the target output
        if self.num == 2:
            out[:, : w // 2] = self.mask_with
        # if num of quadrants to be used as input is 3,
        # also removes the top right quadrant from the target output
        if self.num == 3:
            out[: h // 2, :] = self.mask_with

        # now, sets the input as complementary
        inp = tensor.clone()
        inp[out != -1] = self.mask_with

        sample["input"] = inp
        sample["output"] = out
        return sample


def get_data_CIFAR(num_quadrant_inputs, batch_size):
    """Returns the datasets, dataloaders, and dataset sizes for the CIFAR-10 dataset.

    Args:
        num_quadrant_inputs (int): Number of quadrants to be used as inputs.
        batch_size (int): Batch size for the dataloaders.
        dataset_name (str): Name of the dataset. Defaults to "cifar10".

    Returns:
        datasets (dict): Dictionary containing the train and test datasets.
        dataloaders (dict): Dictionary containing the train and test dataloaders.
        dataset_sizes (dict): Dictionary containing the sizes of the train and test datasets.
    """
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            MaskImages(num_quadrant_inputs=num_quadrant_inputs),
        ]
    )
    train_dataset = CIFAR10(
        train=True,
    )
    test_dataset = CIFAR10(
        train=False,
    )

    train_dataset = MaskImages(num_quadrant_inputs=num_quadrant_inputs)(train_dataset)
    test_dataset = MaskImages(num_quadrant_inputs=num_quadrant_inputs)(test_dataset)

    datasets = {"train": train_dataset, "test": test_dataset}
    dataloaders = {
        "train": DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
        "test": DataLoader(test_dataset, batch_size=batch_size, shuffle=False),
    }
    dataset_sizes = {"train": len(train_dataset), "test": len(test_dataset)}

    return datasets, dataloaders, dataset_sizes