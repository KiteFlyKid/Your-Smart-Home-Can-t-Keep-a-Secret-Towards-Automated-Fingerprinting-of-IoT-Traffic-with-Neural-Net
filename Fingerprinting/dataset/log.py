#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import torch
import pickle
from torch.utils.data import Dataset, Sampler


class smarthome_dataset(Dataset):
    def __init__(self, dir,filename):
        with open(dir+filename,'rb' ) as f:
            self.dataset=pickle.load(f)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        x,y=self.dataset[idx]
        return torch.tensor(x),torch.tensor(y)
        # y=self.dataset.iloc[idx,-1]
        # return x, y


