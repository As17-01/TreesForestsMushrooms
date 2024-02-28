from typing import Optional

import numpy as np
import pandas as pd

from src.node import Node
from src.split_detective import SplitDetective


class BaselineDecisionTreeClassifier:
    def __init__(self, max_depth: int, min_samples_split: int):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split

        self.root_node = None

    def fit(self, x_train: pd.DataFrame, y_train: pd.Series):
        self.root_node = self._build_node(x_train.values, y_train.values, None)

    def predict(self, x_test: pd.DataFrame):
        if self.root_node is None:
            raise ValueError("Train the model first!")

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

        sd = SplitDetective(x_values, y_values)
        best_feature_id, best_split_value = sd.get_best_feature()
        node.split_feature_id = best_feature_id
        node.split_value = best_split_value

        node.left = self._init_new_node(x_values, y_values, node, "left")
        node.right = self._init_new_node(x_values, y_values, node, "right")

        return node

    def _init_new_node(self, x_values: np.ndarray, y_values: np.ndarray, node: Node, type: str):
        new_node = Node()
        new_node.set_height(node.height + 1)

        if type == "left":
            idx = x_values[:, node.split_feature_id] == node.split_value
        else:
            idx = x_values[:, node.split_feature_id] != node.split_value

        x_values = x_values[idx]
        y_values = y_values[idx]

        new_node = self._build_node(x_values, y_values, new_node)
        return new_node
