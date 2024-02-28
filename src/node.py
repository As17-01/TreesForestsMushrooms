import numpy as np


class Node:
    def __init__(self):
        self.height = 1

        # if splitting
        self.split_feature_id = None
        self.split_value = None
        self.left = None
        self.right = None

        # if terminating
        self.score = None

    def set_height(self, height: int):
        self.height = height

    def forward(self, item: np.ndarray):
        if self.left is None and self.right is None:
            return self.score
        if item[self.split_feature_id] == self.split_value:
            return self.left.forward(item)
        else:
            return self.right.forward(item)
