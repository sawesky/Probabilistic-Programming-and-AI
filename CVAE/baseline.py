# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import copy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torchvision.models import resnet18, ResNet18_Weights
#from mxnet import gluon, nd
#from mxnet.gluon.data.vision import transforms
from gluoncv.model_zoo import get_model

class BaselineNet(nn.Module):
    def __init__(self, hidden_1, hidden_2):
        super().__init__()
        self.fc1 = nn.Linear(784, hidden_1)
        self.fc2 = nn.Linear(hidden_1, hidden_2)
        self.fc3 = nn.Linear(hidden_2, 784)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(-1, 784)
        hidden = self.relu(self.fc1(x))
        hidden = self.relu(self.fc2(hidden))
        y = torch.sigmoid(self.fc3(hidden))
        return y


class BaselineNetCIFAR10(nn.Module):
    def __init__(self, hidden_1, hidden_2):
        super().__init__()
        self.fc1 = nn.Linear(3072, hidden_1)  # Input size updated for CIFAR-10
        self.fc2 = nn.Linear(hidden_1, hidden_2)
        self.fc3 = nn.Linear(hidden_2, 3072)  # Output size updated for CIFAR-10
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(-1, 3072)  # Flatten CIFAR-10 images
        hidden = self.relu(self.fc1(x))
        hidden = self.relu(self.fc2(hidden))
        y = torch.sigmoid(self.fc3(hidden))
        return y

class BaselineNetCIFAR10GluonResNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Load a pre-trained GluonCV ResNet model (CIFAR-10 version)
        self.net = get_model('cifar_resnet110_v1', classes=10, pretrained=True)
        # Convert Gluon model to PyTorch-compatible format
        self.net.hybridize()  # This step is essential for running the model with MXNet efficiently

    def forward(self, x):
        # Pre-process the input to match the GluonCV preprocessing pipeline
        #transform_fn = transforms.Compose([
        #    transforms.Resize(32),
        #    transforms.CenterCrop(32),
        #    transforms.ToTensor(),
        #    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
        #])
        #x = transform_fn(x)  # Apply the necessary transformations

        # Feed the transformed image through the model
        x = nd.array(x).unsqueeze(0)  # Add batch dimension for prediction
        pred = self.net(x)

        return torch.sigmoid(torch.tensor(pred))
        
class BaselineNetCIFAR10ResNet18(nn.Module):
    def __init__(self):
        super().__init__()
        # Load a pretrained ResNet model
        self.pretrained = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1) 
        # Modify the final fully connected layer
        num_features = self.pretrained.fc.in_features
        self.pretrained.fc = nn.Linear(num_features, 3072)  # CIFAR-10 output size

    def forward(self, x):
        x = x.view(-1, 3, 32, 32)  # Reshape for ResNet
        y = self.pretrained(x)
        return torch.sigmoid(y)


class MaskedBCELoss(nn.Module):
    def __init__(self, masked_with=-1):
        super().__init__()
        self.masked_with = masked_with

    def forward(self, input, target):
        target = target.view(input.shape)
        # only calculate loss on target pixels (value = -1)
        loss = F.binary_cross_entropy(
            input[target != self.masked_with],
            target[target != self.masked_with],
            reduction="none",
        )
        return loss.sum()

def train(
    device,
    dataloaders,
    dataset_sizes,
    learning_rate,
    num_epochs,
    early_stop_patience,
    model_path,
    dataset,
    hidden_1,
    hidden_2,
    pretrained
):
    criterion = MaskedBCELoss()
    # Train baseline
    if dataset == "mnist":
        baseline_net = BaselineNet(hidden_1, hidden_2)
    elif dataset == "cifar10":
        if pretrained:
            baseline_net = BaselineNetCIFAR10GluonResNet()
        else:
            baseline_net = BaselineNetCIFAR10(hidden_1, hidden_2)
    elif dataset == "fashionmnist":
        baseline_net = BaselineNet(hidden_1, hidden_2)
    else:
        raise ValueError(f"Dataset {dataset} not supported")
    baseline_net.to(device)
    optimizer = torch.optim.Adam(baseline_net.parameters(), lr=learning_rate)
    best_loss = np.inf
    early_stop_count = 0
    for epoch in range(num_epochs):
        for phase in ["train", "val"]:
            if phase == "train":
                baseline_net.train()
            else:
                baseline_net.eval()
    
            running_loss = 0.0
            num_preds = 0
    
            bar = tqdm(
                dataloaders[phase],
                desc="NN Epoch {} {}".format(epoch, phase).ljust(20),
            )
            for i, batch in enumerate(bar):
                inputs = batch["input"].to(device)
                outputs = batch["output"].to(device)
    
                optimizer.zero_grad()
    
                with torch.set_grad_enabled(phase == "train"):
                    preds = baseline_net(inputs)
    
                    # Check for NaNs in the predictions
                    if torch.any(torch.isnan(preds)):
                        print("NaN detected in the predictions")
                    
                    loss = criterion(preds, outputs) / inputs.size(0)
    
                    # Check the loss value
                    if torch.isnan(loss) or torch.isinf(loss):
                        print(f"NaN or Inf detected in loss: {loss.item()}")
    
                    if phase == "train":
                        loss.backward()
                        optimizer.step()
    
                running_loss += loss.item()
                num_preds += 1
                if i % 10 == 0:
                    bar.set_postfix(
                        loss="{:.2f}".format(running_loss / num_preds),
                        early_stop_count=early_stop_count,
                    )
    
            epoch_loss = running_loss / dataset_sizes[phase]
            # deep copy the model
            if phase == "val":
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    best_model_wts = copy.deepcopy(baseline_net.state_dict())
                    early_stop_count = 0
                else:
                    early_stop_count += 1
    
        if early_stop_count >= early_stop_patience:
            break
    
    baseline_net.load_state_dict(best_model_wts)
    baseline_net.eval()
    
    # Save model weights
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(baseline_net.state_dict(), model_path)
    
    return baseline_net
