from typing import Any
from typing import Dict
from typing import Optional

import numpy as np
import pandas as pd

from src.base import BaseModel
from src.node import Node
from src.split_detective import SplitDetective


class BaselineDecisionTreeClassifier(BaseModel):
    def __init__(self, max_depth: int, min_samples_split: int, criterion: str, random_state: Optional[int] = None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion = criterion
        self.random_state = random_state

        self.encoders: Dict[str, Any] = {}
        self.root_node = None

    def fit(self, x_train: pd.DataFrame, y_train: pd.Series):
        x_train = x_train.copy()
        y_train = y_train.copy()
        x_train = self._process_categorical(x=x_train, is_train=True)

        self.root_node = self._build_node(x_train.values, y_train.values, None)

    def predict(self, x_test: pd.DataFrame):
        if self.root_node is None:
            raise ValueError("Train the model first!")

        x_test = x_test.copy()
        x_test = self._process_categorical(x=x_test, is_train=True)

        ans = np.zeros(x_test.shape[0], dtype="float32")
        for i in range(x_test.shape[0]):
            ans[i] = self.root_node.forward(x_test.values[i])
        return ans

    def _build_node(self, x_values: np.ndarray, y_values: np.ndarray, node: Optional[Node]):
        if node is None:
            node = Node()

        # terminating cases
        is_label_the_same = np.unique(y_values).shape[0] == 1
        if is_label_the_same:
            node.score = y_values[0].astype("float32")
            return node

        is_no_more_features = x_values.shape[1] == 0
        is_not_enough_samples = x_values.shape[0] < self.min_samples_split
        is_max_depth = node.height == self.max_depth
        if is_no_more_features or is_not_enough_samples or is_max_depth:
            if len(y_values) > 0:
                node.score = (np.sum(y_values) / len(y_values)).astype("float32")
            else:
                node.score = 0.0
            return node

        sd = SplitDetective(
            x_values=x_values, y_values=y_values, criterion=self.criterion, random_state=self.random_state
        )
        best_feature_id, best_operation = sd.get_best_feature(cat_features_idx=[0, 1, 2])
        node.split_feature_id = best_feature_id
        node.operation = best_operation

        node.true_node = self._init_new_node(x_values, y_values, node, "true_node")
        node.false_node = self._init_new_node(x_values, y_values, node, "false_node")

        return node

    def _init_new_node(self, x_values: np.ndarray, y_values: np.ndarray, node: Node, type: str):
        new_node = Node()
        new_node.set_height(node.height + 1)

        idx = node.operation(x_values[:, node.split_feature_id])
        if type == "false_node":
            idx = ~idx

        x_values = x_values[idx]
        y_values = y_values[idx]

        new_node = self._build_node(x_values, y_values, new_node)
        return new_node
