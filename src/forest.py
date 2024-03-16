from typing import Any
from typing import Dict
from typing import Optional

import numpy as np
import pandas as pd

from src.encoder import LabelEncoder
from src.tree import BaselineDecisionTreeClassifier


class RandomForest:
    def __init__(
        self,
        num_estimators: int,
        max_depth: int,
        min_samples_split: int,
        criterion: str,
        random_state: Optional[int] = None,
    ):
        self.num_estimators = num_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion = criterion
        self.random_state = random_state

        self.encoders: Dict[str, Any] = {}
        # self.drop_features = []
        self.trees = []

    def _process_categorical(self, x: pd.DataFrame, is_train: bool = False) -> pd.DataFrame:
        if is_train:
            self.encoders = {}
            for col_name in x.select_dtypes(include=["object"]):
                encoder = LabelEncoder()

                encoder.fit(x[col_name].values)
                self.encoders[col_name] = encoder

                x[col_name] = encoder.encode(x[col_name].values)
        else:
            for col_name, encoder in self.encoders.items():
                x[col_name] = encoder.encode(x[col_name].values)
        return x

    def fit(self, x_train: pd.DataFrame, y_train: pd.Series):
        x_train = x_train.copy()
        y_train = y_train.copy()
        x_train = self._process_categorical(x=x_train, is_train=True)

        self.trees = []
        for i in range(self.num_estimators):
            np.random.seed(self.random_state * 10 + i * 4)
            train_index = np.random.choice(np.arange(len(x_train)), size=int(len(x_train)), replace=True)

            cur_x_train = x_train.iloc[train_index].copy()
            cur_y_train = y_train.iloc[train_index].copy()

            # n_drop_features = np.random.randint(0, len(cur_x_train.columns) - 1)
            # drop_features = np.random.choice(cur_x_train.columns, n_drop_features, replace=False)

            # cur_x_train.drop(columns=drop_features, inplace=True)
            # self.drop_features.append(drop_features)

            tree = BaselineDecisionTreeClassifier(
                self.max_depth, self.min_samples_split, self.criterion, self.random_state
            )
            tree.fit(cur_x_train, cur_y_train)

            self.trees.append(tree)

    def predict(self, x_test: pd.DataFrame):
        x_test = x_test.copy()
        x_test = self._process_categorical(x=x_test, is_train=True)

        ans = np.zeros(x_test.shape[0], dtype="float32")
        for tree in self.trees:
            # cur_x_test = x_test.drop(columns=drop_features)
            for i in range(x_test.shape[0]):
                ans[i] += tree.root_node.forward(x_test.values[i]) / len(self.trees)
        return ans
