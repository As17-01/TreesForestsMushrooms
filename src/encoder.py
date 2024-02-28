from typing import Dict
import pandas as pd
import numpy as np

class LabelEncoder:
    def __init__(self):
        self.mapping: Dict[str, int] = {}

    def fit(self, feature: np.ndarray) -> None:
        labels = np.unique(feature)

        for i, cur_l in enumerate(labels):
            self.mapping[cur_l] = i

    def encode(self, feature: np.ndarray) -> np.ndarray:
        encoded_feature = np.zeros(len(feature), dtype="float64")
        for i, val in enumerate(feature):
            encoded_feature[i] = self.mapping[val]

        return encoded_feature
