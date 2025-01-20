# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO


class EncoderCIFAR10(nn.Module):
    def __init__(self, z_dim, hidden_1, hidden_2):
        super().__init__()
        self.fc1 = nn.Linear(3072, hidden_1)  # Adjusted input size for CIFAR-10
        self.fc2 = nn.Linear(hidden_1, hidden_2)
        self.fc31 = nn.Linear(hidden_2, z_dim)
        self.fc32 = nn.Linear(hidden_2, z_dim)
        self.relu = nn.ReLU()

    def forward(self, x, y):
        # Combine x and y for processing
        xc = x.clone()
        xc[x == -1] = y[x == -1]
        xc = xc.view(-1, 3072)  # Flatten CIFAR-10 images
        hidden = self.relu(self.fc1(xc))
        hidden = self.relu(self.fc2(hidden))
        z_loc = self.fc31(hidden)
        z_scale = torch.exp(self.fc32(hidden))
        return z_loc, z_scale

class EncoderCIFAR10ResNet18(nn.Module):
    def __init__(self, z_dim, hidden_1, hidden_2, pretrained_model):
        super().__init__()
        self.pretrained = pretrained_model
        self.fc1 = nn.Linear(3072, hidden_2)  # Use output from pretrained model
        self.fc31 = nn.Linear(hidden_2, z_dim)
        self.fc32 = nn.Linear(hidden_2, z_dim)
        self.relu = nn.ReLU()

    def forward(self, x, y):
        # Combine x and y
        xc = x.clone()
        xc[x == -1] = y[x == -1]
        xc = xc.view(-1, 3, 32, 32)  # Reshape for pretrained model
        features = self.pretrained(xc)
        hidden = self.relu(self.fc1(features))
        z_loc = self.fc31(hidden)
        z_scale = torch.exp(self.fc32(hidden))
        return z_loc, z_scale

class DecoderCIFAR10(nn.Module):
    def __init__(self, z_dim, hidden_1, hidden_2):
        super().__init__()
        self.fc1 = nn.Linear(z_dim, hidden_2)
        self.fc2 = nn.Linear(hidden_2, hidden_1)
        self.fc3 = nn.Linear(
            hidden_1, 3072
        )  # Adjusted output size for CIFAR-10
        self.relu = nn.ReLU()

    def forward(self, z):
        y = self.relu(self.fc1(z))
        y = self.relu(self.fc2(y))
        y = torch.sigmoid(self.fc3(y))
        return y.view(-1, 3, 32, 32)  # Reshape to CIFAR-10 dimensions


class EncoderCIFAR10Conv(nn.Module):
    def __init__(self, z_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.fc1 = nn.Linear(128 * 4 * 4, z_dim)
        self.fc2 = nn.Linear(128 * 4 * 4, z_dim)
        self.relu = nn.ReLU()

    def forward(self, x, y):
        # Combine x and y for processing
        xc = x.clone()
        xc[x == -1] = y[x == -1]
        xc = self.relu(self.bn1(self.conv1(xc)))
        xc = self.relu(self.bn2(self.conv2(xc)))
        xc = self.relu(self.bn3(self.conv3(xc)))
        xc = xc.view(xc.size(0), -1)  # Flatten
        z_loc = self.fc1(xc)
        z_scale = torch.exp(self.fc2(xc))
        return z_loc, z_scale


class DecoderCIFAR10Conv(nn.Module):
    def __init__(self, z_dim):
        super().__init__()
        self.fc = nn.Linear(z_dim, 128 * 4 * 4)
        self.deconv1 = nn.ConvTranspose2d(
            128, 64, kernel_size=4, stride=2, padding=1
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.deconv2 = nn.ConvTranspose2d(
            64, 32, kernel_size=4, stride=2, padding=1
        )
        self.bn2 = nn.BatchNorm2d(32)
        self.deconv3 = nn.ConvTranspose2d(
            32, 3, kernel_size=4, stride=2, padding=1
        )
        self.bn3 = nn.BatchNorm2d(3)
        self.relu = nn.ReLU()

    def forward(self, z):
        y = self.relu(self.fc(z))
        y = y.view(
            y.size(0), 128, 4, 4
        )  # Reshape to match the dimensions before deconvolution
        y = self.relu(self.bn1(self.deconv1(y)))
        y = self.relu(self.bn2(self.deconv2(y)))
        y = torch.sigmoid(self.bn3(self.deconv3(y)))
        return y


class Encoder(nn.Module):
    def __init__(self, z_dim, hidden_1, hidden_2):
        super().__init__()
        self.fc1 = nn.Linear(784, hidden_1)
        self.fc2 = nn.Linear(hidden_1, hidden_2)
        self.fc31 = nn.Linear(hidden_2, z_dim)
        self.fc32 = nn.Linear(hidden_2, z_dim)
        self.relu = nn.ReLU()

    def forward(self, x, y):
        # put x and y together in the same image for simplification
        xc = x.clone()
        xc[x == -1] = y[x == -1]
        xc = xc.view(-1, 784)
        # then compute the hidden units
        hidden = self.relu(self.fc1(xc))
        hidden = self.relu(self.fc2(hidden))
        # then return a mean vector and a (positive) square root covariance
        # each of size batch_size x z_dim
        z_loc = self.fc31(hidden)
        z_scale = torch.exp(self.fc32(hidden))
        return z_loc, z_scale


class Decoder(nn.Module):
    def __init__(self, z_dim, hidden_1, hidden_2):
        super().__init__()
        self.fc1 = nn.Linear(z_dim, hidden_1)
        self.fc2 = nn.Linear(hidden_1, hidden_2)
        self.fc3 = nn.Linear(hidden_2, 784)
        self.relu = nn.ReLU()

    def forward(self, z):
        y = self.relu(self.fc1(z))
        y = self.relu(self.fc2(y))
        y = torch.sigmoid(self.fc3(y))
        return y


class CVAECIFAR(nn.Module):
    def __init__(self, z_dim, hidden_1, hidden_2, pre_trained_baseline_net):
        super().__init__()
        self.baseline_net = pre_trained_baseline_net
        self.prior_net = EncoderCIFAR10(z_dim, hidden_1, hidden_2)
        self.generation_net = DecoderCIFAR10(z_dim, hidden_1, hidden_2)
        self.recognition_net = EncoderCIFAR10(z_dim, hidden_1, hidden_2)

    def model(self, xs, ys=None):
        pyro.module("generation_net", self)
        batch_size = xs.shape[0]
        with pyro.plate("data"):
            # Generate initial guess using baseline network
            with torch.no_grad():
                y_hat = self.baseline_net(xs).view(batch_size, 3, 32, 32)

            # Sample latent variable z from prior
            prior_loc, prior_scale = self.prior_net(xs, y_hat)
            zs = pyro.sample(
                "z", dist.Normal(prior_loc, prior_scale).to_event(1)
            )

            # Generate output image loc from z
            loc = self.generation_net(zs)

            if ys is not None:
                # In training, only evaluate loss on masked pixels
                mask_loc = loc[xs == -1].view(batch_size, -1)
                mask_ys = ys[xs == -1].view(batch_size, -1)
                pyro.sample(
                    "y",
                    dist.Bernoulli(mask_loc, validate_args=False).to_event(1),
                    obs=mask_ys,
                )
            else:
                # In testing, return probabilities for visualization
                pyro.deterministic("y", loc.detach())

            return loc

    def guide(self, xs, ys=None):
        with pyro.plate("data"):
            if ys is None:
                # Inference uses the prior network
                y_hat = self.baseline_net(xs).view(xs.shape)
                loc, scale = self.prior_net(xs, y_hat)
            else:
                # Training uses the recognition network
                loc, scale = self.recognition_net(xs, ys)

            pyro.sample("z", dist.Normal(loc, scale).to_event(1))

class CVAECIFAR10ResNet18(nn.Module):
    def __init__(self, z_dim, hidden_1, hidden_2, pre_trained_baseline_net):
        super().__init__()
        self.baseline_net = pre_trained_baseline_net
        self.prior_net = EncoderCIFAR10ResNet18(z_dim, hidden_1, hidden_2, pre_trained_baseline_net)
        self.generation_net = DecoderCIFAR10(z_dim, hidden_1, hidden_2)
        self.recognition_net = EncoderCIFAR10(z_dim, hidden_1, hidden_2)

    def model(self, xs, ys=None):
        pyro.module("generation_net", self)
        batch_size = xs.shape[0]
        with pyro.plate("data"):
            # Generate initial guess using baseline network
            with torch.no_grad():
                y_hat = self.baseline_net(xs).view(batch_size, 3, 32, 32)

            # Sample latent variable z from prior
            prior_loc, prior_scale = self.prior_net(xs, y_hat)
            zs = pyro.sample(
                "z", dist.Normal(prior_loc, prior_scale).to_event(1)
            )

            # Generate output image loc from z
            loc = self.generation_net(zs)

            if ys is not None:
                # In training, only evaluate loss on masked pixels
                mask_loc = loc[xs == -1].view(batch_size, -1)
                mask_ys = ys[xs == -1].view(batch_size, -1)
                pyro.sample(
                    "y",
                    dist.Bernoulli(mask_loc, validate_args=False).to_event(1),
                    obs=mask_ys,
                )
            else:
                # In testing, return probabilities for visualization
                pyro.deterministic("y", loc.detach())

            return loc

    def guide(self, xs, ys=None):
        with pyro.plate("data"):
            if ys is None:
                # Inference uses the prior network
                y_hat = self.baseline_net(xs).view(xs.shape)
                loc, scale = self.prior_net(xs, y_hat)
            else:
                # Training uses the recognition network
                loc, scale = self.recognition_net(xs, ys)

            pyro.sample("z", dist.Normal(loc, scale).to_event(1))



class CVAECIFARConv(nn.Module):
    def __init__(self, z_dim, pre_trained_baseline_net):
        super().__init__()
        self.baseline_net = pre_trained_baseline_net
        self.prior_net = EncoderCIFAR10Conv(z_dim)
        self.generation_net = DecoderCIFAR10Conv(z_dim)
        self.recognition_net = EncoderCIFAR10Conv(z_dim)

    def model(self, xs, ys=None):
        pyro.module("generation_net", self)
        batch_size = xs.shape[0]
        with pyro.plate("data"):
            # Generate initial guess using baseline network
            with torch.no_grad():
                y_hat = self.baseline_net(xs).view(batch_size, 3, 32, 32)

            # Sample latent variable z from prior
            prior_loc, prior_scale = self.prior_net(xs, y_hat)
            zs = pyro.sample(
                "z", dist.Normal(prior_loc, prior_scale).to_event(1)
            )

            # Generate output image loc from z
            loc = self.generation_net(zs)

            if ys is not None:
                # In training, only evaluate loss on masked pixels
                mask_loc = loc[xs == -1].view(batch_size, -1)
                mask_ys = ys[xs == -1].view(batch_size, -1)
                pyro.sample(
                    "y",
                    dist.Bernoulli(mask_loc, validate_args=False).to_event(1),
                    obs=mask_ys,
                )
            else:
                # In testing, return probabilities for visualization
                pyro.deterministic("y", loc.detach())

            return loc

    def guide(self, xs, ys=None):
        with pyro.plate("data"):
            if ys is None:
                # Inference uses the prior network
                y_hat = self.baseline_net(xs).view(xs.shape)
                loc, scale = self.prior_net(xs, y_hat)
            else:
                # Training uses the recognition network
                loc, scale = self.recognition_net(xs, ys)

            pyro.sample("z", dist.Normal(loc, scale).to_event(1))


class CVAE(nn.Module):
    def __init__(self, z_dim, hidden_1, hidden_2, pre_trained_baseline_net):
        super().__init__()
        # The CVAE is composed of multiple MLPs, such as recognition network
        # qφ(z|x, y), (conditional) prior network pθ(z|x), and generation
        # network pθ(y|x, z). Also, CVAE is built on top of the NN: not only
        # the direct input x, but also the initial guess y_hat made by the NN
        # are fed into the prior network.
        self.baseline_net = pre_trained_baseline_net
        self.prior_net = Encoder(z_dim, hidden_1, hidden_2)
        self.generation_net = Decoder(z_dim, hidden_1, hidden_2)
        self.recognition_net = Encoder(z_dim, hidden_1, hidden_2)

    def model(self, xs, ys=None):
        # register this pytorch module and all of its sub-modules with pyro
        pyro.module("generation_net", self)
        batch_size = xs.shape[0]
        with pyro.plate("data"):
            # Prior network uses the baseline predictions as initial guess.
            # This is the generative process with recurrent connection
            with torch.no_grad():
                # this ensures the training process does not change the
                # baseline network
                y_hat = self.baseline_net(xs).view(xs.shape)

            # sample the handwriting style from the prior distribution, which is
            # modulated by the input xs.
            prior_loc, prior_scale = self.prior_net(xs, y_hat)
            zs = pyro.sample(
                "z", dist.Normal(prior_loc, prior_scale).to_event(1)
            )

            # the output y is generated from the distribution pθ(y|x, z)
            loc = self.generation_net(zs)

            if ys is not None:
                # In training, we will only sample in the masked image
                mask_loc = loc[(xs == -1).view(-1, 784)].view(batch_size, -1)
                mask_ys = ys[xs == -1].view(batch_size, -1)
                pyro.sample(
                    "y",
                    dist.Bernoulli(mask_loc, validate_args=False).to_event(1),
                    obs=mask_ys,
                )
            else:
                # In testing, no need to sample: the output is already a
                # probability in [0, 1] range, which better represent pixel
                # values considering grayscale. If we sample, we will force
                # each pixel to be  either 0 or 1, killing the grayscale
                pyro.deterministic("y", loc.detach())

            # return the loc so we can visualize it later
            return loc

    def guide(self, xs, ys=None):
        with pyro.plate("data"):
            if ys is None:
                # at inference time, ys is not provided. In that case,
                # the model uses the prior network
                y_hat = self.baseline_net(xs).view(xs.shape)
                loc, scale = self.prior_net(xs, y_hat)
            else:
                # at training time, uses the variational distribution
                # q(z|x,y) = normal(loc(x,y),scale(x,y))
                loc, scale = self.recognition_net(xs, ys)

            pyro.sample("z", dist.Normal(loc, scale).to_event(1))


def train(
    device,
    dataloaders,
    dataset_sizes,
    learning_rate,
    num_epochs,
    early_stop_patience,
    model_path,
    pre_trained_baseline_net,
    dataset,
    z_dim,
    hidden_1,
    hidden_2,
    use_conv,
    random_mask,
    pretrained
):
    # clear param store
    pyro.clear_param_store()

    if dataset == "mnist":
        cvae_net = CVAE(z_dim, hidden_1, hidden_2, pre_trained_baseline_net)
    elif dataset == "cifar10":
        if use_conv:
            cvae_net = CVAECIFARConv(z_dim, pre_trained_baseline_net)
        else:
            if pretrained:
                cvae_net = CVAECIFAR10ResNet18(z_dim, hidden_1, hidden_2, pre_trained_baseline_net)
            else:
                cvae_net = CVAECIFAR(
                    z_dim, hidden_1, hidden_2, pre_trained_baseline_net
                )
    elif dataset == "fashionmnist":
        cvae_net = CVAE(z_dim, hidden_1, hidden_2, pre_trained_baseline_net)
    else:
        raise ValueError("Dataset not supported")
    cvae_net.to(device)
    optimizer = pyro.optim.Adam({"lr": learning_rate})
    svi = SVI(cvae_net.model, cvae_net.guide, optimizer, loss=Trace_ELBO())

    best_loss = np.inf
    early_stop_count = 0
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(num_epochs):
        # Each epoch has a training and validation phase
        for phase in ["train", "val"]:
            running_loss = 0.0
            num_preds = 0

            # Iterate over data.
            bar = tqdm(
                dataloaders[phase],
                desc="CVAE Epoch {} {}".format(epoch, phase).ljust(20),
            )
            for i, batch in enumerate(bar):
                inputs = batch["input"].to(device)
                outputs = batch["output"].to(device)

                if phase == "train":
                    loss = svi.step(inputs, outputs)
                else:
                    loss = svi.evaluate_loss(inputs, outputs)

                # statistics
                running_loss += loss / inputs.size(0)
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
                    torch.save(cvae_net.state_dict(), model_path)
                    early_stop_count = 0
                else:
                    early_stop_count += 1

        if early_stop_count >= early_stop_patience:
            break

    # Save model weights
    cvae_net.load_state_dict(torch.load(model_path, weights_only=False))
    cvae_net.eval()
    return cvae_net
