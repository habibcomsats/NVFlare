# Copyright (c) 2021, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import torch
from torchvision import datasets


class BasicDataset(torch.utils.data.Dataset):
    def __init__(self, data=None, targets=None, transform=None, download=False):
        """Basic torch Dataset.
        To be used with CIFAR-10 data and targets, For example.

        Args:
            data: input data
            targets: targets
            transform: image transforms
            download: whether to download the data (default: False)
        Returns:
            A PyTorch dataset
        """
        self.data = data
        self.targets = targets
        self.transform = transform
        self.download = download

        if len(self.data) != len(self.targets):
            raise ValueError(f"Data and targets need have the same length but are {len(self.data)} and {len(self.targets)}.")

    def __getitem__(self, index):
        img, targets = self.data[index], self.targets[index]
        if self.transform is not None:
            img = self.transform(img)
        return img, targets

    def __len__(self):
        return len(self.data)
