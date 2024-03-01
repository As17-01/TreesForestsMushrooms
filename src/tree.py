from typing import Any
from typing import Dict
from typing import Optional

import numpy as np
import pandas as pd

from src.encoder import LabelEncoder
from src.node import Node
from src.split_detective import SplitDetective


class BaselineDecisionTreeClassifier:
    def __init__(self, max_depth: int, min_samples_split: int):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split

        self.encoders: Dict[str, Any] = {}
        self.root_node = None

    def fit(self, x_train: pd.DataFrame, y_train: pd.Series):
        x_train = x_train.copy()
        y_train = y_train.copy()

        self.encoders = {}
        self.root_node = None

        for col_name in x_train.select_dtypes(include=["object"]):
            encoder = LabelEncoder()

            encoder.fit(x_train[col_name].values)
            self.encoders[col_name] = encoder

            x_train[col_name] = encoder.encode(x_train[col_name].values)

        self.root_node = self._build_node(x_train.values, y_train.values, None)

    def predict(self, x_test: pd.DataFrame):
        x_test = x_test.copy()

        for col_name, encoder in self.encoders.items():
            x_test[col_name] = encoder.encode(x_test[col_name].values)

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
        best_feature_id, best_operation = sd.get_best_feature()
        node.split_feature_id = best_feature_id
        node.operation = best_operation

        node.true_node = self._init_new_node(x_values, y_values, node, "true_node")
        node.false_node = self._init_new_node(x_values, y_values, node, "false_node")

        return node

    def _init_new_node(self, x_values: np.ndarray, y_values: np.ndarray, node: Node, type: str):
        new_node = Node()
        new_node.set_height(node.height + 1)

        if type == "true_node":
            idx = [node.operation(val) for val in x_values[:, node.split_feature_id]]
        else:
            idx = [~node.operation(val) for val in x_values[:, node.split_feature_id]]

        x_values = x_values[idx]
        y_values = y_values[idx]

        new_node = self._build_node(x_values, y_values, new_node)
        return new_node
