from typing import Any
from typing import Dict
from typing import List
from typing import Optional

import numpy as np
import pandas as pd

from src.base import BaseModel
from src.tree import BaselineDecisionTreeClassifier


class AdaBoost(BaseModel):
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
        self.weights: List[float] = []
        self.trees: List[Any] = []

    def fit(self, x_train: pd.DataFrame, y_train: pd.Series):
        x_train = x_train.copy()
        y_train = y_train.copy()
        x_train = self._process_categorical(x=x_train, is_train=True)

        self.trees = []
        self.weights = []
        sample_weights = np.ones(len(y_train)) / len(y_train)

        for i in range(self.num_estimators):
            if self.random_state is not None:
                np.random.seed(self.random_state * 10 + i * 4)
            train_index = np.random.choice(np.arange(len(x_train)), size=int(len(x_train)), replace=True)

            cur_x_train = x_train.iloc[train_index].copy()
            cur_y_train = y_train.iloc[train_index].copy()

            tree = BaselineDecisionTreeClassifier(
                self.max_depth, self.min_samples_split, self.criterion, self.random_state + i
            )
            tree.fit(cur_x_train, cur_y_train)
            pred = tree.predict(cur_x_train)
            binarized_pred = np.where(pred > 0.5, 1, 0)

            error_rate = np.sum(binarized_pred != y_train) / len(y_train)
            alpha = np.log((1 - error_rate) / (error_rate + 1e-10)) / 2

            sample_weights = sample_weights * np.exp(-alpha * y_train * binarized_pred)

            self.trees.append(tree)
            self.weights.append(alpha)

            sample_weights = sample_weights / sample_weights.sum()

    def predict(self, x_test: pd.DataFrame):
        x_test = x_test.copy()
        x_test = self._process_categorical(x=x_test, is_train=True)

        ans = np.zeros(x_test.shape[0], dtype="float32")
        for i in range(x_test.shape[0]):
            weighted_sum = 0
            for tree, w in zip(self.trees, self.weights):
                pred = tree.root_node.forward(x_test.values[i])

                pred = np.where(pred > 0.5, 1, -1)
                weighted_sum += w * pred
            ans[i] = np.where(weighted_sum > 0, 1, 0)
        return ans
