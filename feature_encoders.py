import numpy as np
import pandas as pd
import torch
import re
from sentence_transformers import SentenceTransformer

from enum import Enum


class ColType(Enum):
    NUMERICAL = 0
    LARGE_NUMBER = 1
    NUMBER_TO_BIN = 2
    INDEX = 3
    CATEGORY = 4
    STRING = 5
    CLASSIFICATION_TARGET = 6


class StringEncoder(object):
    def __init__(self, model_name='all-MiniLM-L6-v2', device=None):
        self.device = device
        self.model = SentenceTransformer(model_name, device=device)

    @torch.no_grad()
    def __call__(self, df):
        x = self.model.encode(df.dropna().values, show_progress_bar=True,
                              convert_to_tensor=True, device=self.device)
        x = x.cpu()
        res = torch.zeros(len(df), x.size(dim=1))
        res[df.notnull().values, :] = x
        return res


class NumericalEncoder(object):
    def __init__(self):
        return

    def __call__(self, df, na_val=0):
        x = torch.tensor(df.fillna(value=na_val).values)
        return x


class LargeNumberEncoder(object):
    def __init__(self):
        return

    def __call__(self, df, na_val=0):
        x = torch.tensor(df.fillna(value=na_val).values)
        x = (x - torch.min(x)) / (torch.max(x) - torch.min(x))
        return x


class BinEncoder(object):
    def __init__(self):
        return

    def __call__(self, df, num_bins=10):
        cut = pd.qcut(df.astype('float').rank(method='first'), min(num_bins, int(len(df.unique())/2.0)))
        genres = set(genre for genre in cut.values)
        # keep NaN as a bin
        genres_mapping = {genre: i for i, genre in enumerate(genres)}

        x = torch.zeros(len(df), len(genres))
        for i, genre in enumerate(cut.values):
            x[i, genres_mapping[genre]] = 1
        # print(torch.sum(torch.sum(x,dim=1)!=1))
        return x


class IndexEncoder(object):
    def __init__(self):
        return

    def __call__(self, df):
        mapping = {index: i for i, index in enumerate(df.astype('int64').values)}
        return mapping


class CategoryEncoder(object):
    def __init__(self):
        return

    def __call__(self, df):
        genres = set(g for col in df.dropna().values for g in re.split('[^a-zA-Z0-9 ]', col))
        genres.discard('')
        genres_mapping = {genre: i for i, genre in enumerate(genres)}

        x = torch.zeros(len(df), len(genres))
        for i, col in enumerate(df.dropna().values):
            for genre in re.split('[^a-zA-Z0-9 ]', col):
                if genre == '':
                    continue
                x[i, genres_mapping[genre]] = 1
        return x
