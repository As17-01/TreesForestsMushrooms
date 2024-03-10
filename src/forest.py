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
        self.trees = []

    def fit(self, x_train: pd.DataFrame, y_train: pd.Series):
        x_train = x_train.copy()
        y_train = y_train.copy()

        self.encoders = {}
        self.trees = []

        for col_name in x_train.select_dtypes(include=["object"]):
            encoder = LabelEncoder()

            encoder.fit(x_train[col_name].values)
            self.encoders[col_name] = encoder

            x_train[col_name] = encoder.encode(x_train[col_name].values)

        for i in range(self.num_estimators):
            np.random.seed(self.random_state * 10 + i * 4)
            train_index = np.random.choice(np.array(x_train.index), size=int(0.80 * len(x_train)), replace=False)

            is_train = x_train.index.isin(train_index)
            cur_x_train = x_train.iloc[is_train]
            cur_y_train = y_train.iloc[is_train]

            tree = BaselineDecisionTreeClassifier(
                self.max_depth, self.min_samples_split, self.criterion, self.random_state
            )
            tree.fit(cur_x_train, cur_y_train)

            self.trees.append(tree)

    def predict(self, x_test: pd.DataFrame):
        x_test = x_test.copy()

        for col_name, encoder in self.encoders.items():
            x_test[col_name] = encoder.encode(x_test[col_name].values)

        if len(self.trees) == 0:
            raise ValueError("Train the model first!")

        ans = np.zeros(x_test.shape[0], dtype="float32")
        for tree in self.trees:
            for i in range(x_test.shape[0]):
                ans[i] += tree.root_node.forward(x_test.values[i]) / len(self.trees)
        return ans
