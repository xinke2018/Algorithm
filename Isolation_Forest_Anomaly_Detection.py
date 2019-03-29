# Follows algo from https://cs.nju.edu.cn/zhouzh/zhouzh.files/publication/icdm08b.pdf

import numpy as np
from sklearn.metrics import confusion_matrix
import pandas as pd

def compute_c(sample_size):
    """
    Compute c value given the number of instances.
    """
    if sample_size > 2:
        h = np.log(sample_size - 1.0) + 0.5772156649
        return 2.0*h - 2.0* (sample_size - 1.0) / sample_size
    elif sample_size == 2:
        return 1.0
    else:
        return 0.0


def predict_from_anomaly_scores(scores, threshold):
    pred = np.zeros(scores.shape)
    for i in range(scores.size):
        if scores[i] >= threshold:
            pred[i] = 1
        else:
            pred[i] = 0
    return pred


class IsolationTreeEnsemble:
    def __init__(self, sample_size, n_trees=10):
        self.trees = []
        self.n_trees = n_trees
        self.sample_size = sample_size


    def fit(self, X:np.ndarray, improved=False):
        """
        Given a 2D matrix of observations, create an ensemble of IsolationTree
        objects and store them in a list: self.trees.  Convert DataFrames to
        ndarray objects.
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        height_limit = int(np.ceil(np.log2(self.sample_size)))
        idx_arr = np.arange(len(X))
        for tree_idx in range(self.n_trees):
            tree = IsolationTree(height_limit)
            # Sample self.sample_size observations.
            # Use the following code instead of np.random.choice to reduce the number of reshuffles.
            start_idx = (tree_idx * self.sample_size) % len(X)
            if start_idx + self.sample_size >= len(X):
                np.random.shuffle(idx_arr)
            sampled_X = X[idx_arr[start_idx:start_idx+self.sample_size]]
            tree.fit(sampled_X, improved)
            self.trees.append(tree)

        return self


    def path_length(self, X:np.ndarray) -> np.ndarray:
        """
        Given a 2D matrix of observations, X, compute the average path length
        for each observation in X.  Compute the path length for x_i using every
        tree in self.trees then compute the average for each x_i.  Return an
        ndarray of shape (len(X),1).
        """
        if isinstance(X, pd.DataFrame):
            X = X.values

        avg_lengths = np.zeros(len(X))
        for i in range(len(X)):
            total_length = 0
            for tree in self.trees:
                total_length += tree.path_length(X[i])
            avg_lengths[i] = total_length / len(self.trees)
        return avg_lengths


    def anomaly_score(self, X:np.ndarray) -> np.ndarray:
        """
        Given a 2D matrix of observations, X, compute the anomaly score
        for each x_i observation, returning an ndarray of them.
        """
        if isinstance(X, pd.DataFrame):
            X = X.values

        avg_path_lengths = self.path_length(X)
        c_value = compute_c(self.sample_size)
        scores = np.zeros(len(X))
        for i in range(len(avg_path_lengths)):
            scores[i] = 2.0**(-avg_path_lengths[i] / c_value)
        return scores


    def predict_from_anomaly_scores(self, scores:np.ndarray, threshold:float) -> np.ndarray:
        """
        Given an array of scores and a score threshold, return an array of
        the predictions: 1 for any score >= the threshold and 0 otherwise.
        """
        return predict_from_anomaly_scores(scores, threshold)


    def predict(self, X:np.ndarray, threshold:float) -> np.ndarray:
        "A shorthand for calling anomaly_score() and predict_from_anomaly_scores()."
        return self.predict_from_anomaly_scores(self.anomaly_score(X), threshold)


# Return the root of the subtree and the number of nodes in the subtree.
def generate_subtree(X, current_height, height_limit, improved):
    if current_height >= height_limit or len(X) <= 1:
        return TreeNode(left=None, 
                right=None, 
                split_att=None, 
                split_value=None, 
                path_length=compute_c(len(X))+current_height), 1
    else:
        while True:
            att = np.random.randint(len(X[0]))
            min_val = X[:,att].min()
            max_val = X[:,att].max()
            split_val = np.random.uniform(min_val,max_val)
            w = np.where(X[:,att] < split_val,True,False)
            X_l = X[w]
            X_r = X[~w]
            # If improved == true, make the split uneven to reduce the number of nodes.
            # This seems to make the algorithm less sensitive to noise.
            if not improved or len(X) < 10 or len(X_l) < 0.3*len(X) or len(X_l) > 0.7*len(X):
                break
        left_node, left_size = generate_subtree(X_l, current_height+1, height_limit, improved)
        right_node, right_size = generate_subtree(X_r, current_height+1, height_limit, improved)
        return TreeNode(left=left_node, 
                right=right_node, 
                split_att=att, 
                split_value=split_val,
                path_length=None), left_size+right_size+1


class IsolationTree:
    def __init__(self, height_limit):
        self.n_nodes = 0
        self.height_limit = height_limit
 

    def fit(self, X:np.ndarray, improved=False):
        """
        Given a 2D matrix of observations, create an isolation tree. Set field
        self.root to the root of that tree and return it.

        If you are working on an improved algorithm, check parameter "improved"
        and switch to your new functionality else fall back on your original code.
        """
        self.root, self.n_nodes = generate_subtree(X, 0, self.height_limit, improved)
        return self.root


    def path_length(self, x:np.ndarray):
        """
        Given an observation, x, compute its path length. 
        """
        current_node = self.root
        while True:
            if current_node.is_external():
                return current_node.path_length
            if x[current_node.split_att] < current_node.split_value:
                current_node = current_node.left
            else:
                current_node = current_node.right


class TreeNode:
    def __init__(self, left, right, split_att, split_value, path_length):
        self.left = left
        self.right = right
        self.split_att = split_att
        self.split_value = split_value
        self.path_length = path_length


    def is_external(self):
        return self.left == None and self.right == None


def find_TPR_threshold(y, scores, desired_TPR):
    """
    Start at score threshold 1.0 and work down until we hit desired TPR.
    Step by 0.01 score increments. For each threshold, compute the TPR
    and FPR to see if we've reached to the desired TPR. If so, return the
    score threshold and FPR.
    """
    threshold = 1.0
    # Only use isolation tree to predict from anomaly scores, and therefore
    # the selection of sample size doesn't matter.
    while threshold >= 0:
        y_pred = predict_from_anomaly_scores(scores, threshold=threshold)
        confusion = confusion_matrix(y, y_pred)
        TN, FP, FN, TP = confusion.flat
        TPR = TP / (TP + FN)
        FPR = FP / (FP + TN)
        if TPR >= desired_TPR:
            return threshold, FPR
        threshold -= 0.01

