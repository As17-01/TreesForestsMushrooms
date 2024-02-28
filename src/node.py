import numpy as np


class Node:
    def __init__(self):
        self.height = 1

        # if splitting
        self.split_feature_id = None
        self.operation = None
        self.true_node = None
        self.false_node = None

        # if terminating
        self.score = None

    def set_height(self, height: int):
        self.height = height

    def forward(self, item: np.ndarray):
        if self.true_node is None and self.false_node is None:
            return self.score
        if self.operation(item[self.split_feature_id]):
            return self.true_node.forward(item)
        else:
            return self.false_node.forward(item)
