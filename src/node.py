class Node:
    def __init__(self):
        self.type = None
        self.height = 1
        self.entropy = None
        self.n_samples = None

        # if splitting
        self.IG = None
        self.split_feature_id = None
        self.split_feature = None
        self.split_value = None
        self.left = None
        self.right = None

        # if terminating
        self.label = None
