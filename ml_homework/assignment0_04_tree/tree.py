import numpy as np
from sklearn.base import BaseEstimator
# from tqdm.notebook import tqdm, trange


def get_freq_elem(arr):
    """
    Computes the most frequent elem with the smallest value
    
    Parameters
    ----------
    arr : np.array of type int
    
    Returns
    -------
    int
        The most frequent elem in the provided subset
    """
    # print('Get_freq_elem arr length:', arr.size)
    arr = np.ravel(arr)
    values, counts = np.unique(arr, return_counts=True)
    return values[counts == counts.max()].min()


def get_classes_proba(arr):
    """
    Computes the frequency of each class in arr
    
    Parameters
    ----------
    arr : np.array of type float with shape (n_objects, n_classes)
        One-hot representation of class labels for corresponding subset
    
    Returns
    -------
    np.array of type float with shape (n_classes, )
        Frequency of each class in the provided subset
    """
    assert np.sum(arr) > 0
    return np.sum(arr, axis=0) / np.sum(arr)
    

def entropy(y):  
    """
    Computes entropy of the provided distribution. Use log(value + eps) for numerical stability
    
    Parameters
    ----------
    y : np.array of type float with shape (n_objects, n_classes)
        One-hot representation of class labels for corresponding subset
    
    Returns
    -------
    float
        Entropy of the provided subset
    """
    EPS = 0.0005

    classes_proba = get_classes_proba(y)
    
    return -np.dot(classes_proba, np.log(EPS + classes_proba))
    
def gini(y):
    """
    Computes the Gini impurity of the provided distribution
    
    Parameters
    ----------
    y : np.array of type float with shape (n_objects, n_classes)
        One-hot representation of class labels for corresponding subset
    
    Returns
    -------
    float
        Gini impurity of the provided subset
    """

    classes_proba = get_classes_proba(y)
    
    return 1 - np.sum(np.square(classes_proba))
    
def variance(y):
    """
    Computes the variance the provided target values subset
    
    Parameters
    ----------
    y : np.array of type float with shape (n_objects, 1)
        Target values vector
    
    Returns
    -------
    float
        Variance of the provided target vector
    """
    
    return y.var()

def mad_median(y):
    """
    Computes the mean absolute deviation from the median in the
    provided target values subset
    
    Parameters
    ----------
    y : np.array of type float with shape (n_objects, 1)
        Target values vector
    
    Returns
    -------
    float
        Mean absolute deviation from the median in the provided vector
    """
    
    y_med = np.median(y)
    
    return np.mean(np.abs(y - y_med))


def one_hot_encode(n_classes, y):
    y_one_hot = np.zeros((len(y), n_classes), dtype=float)
    y_one_hot[np.arange(len(y)), y.astype(int)[:, 0]] = 1.
    return y_one_hot


def one_hot_decode(y_one_hot):
    return y_one_hot.argmax(axis=1)[:, None]


class Node:
    """
    This class is provided "as is" and it is not mandatory to it use in your code.
    """
    def __init__(self, feature_index, threshold, proba=0):
        self.feature_index = feature_index
        self.value = threshold
        self.proba = proba
        self.left_child = None
        self.right_child = None
        
        
class DecisionTree(BaseEstimator):
    DEF_THR = None
    DEF_FEAT_IND = -1
    
    all_criterions = {
        'gini': (gini, True), # (criterion, classification flag)
        'entropy': (entropy, True),
        'variance': (variance, False),
        'mad_median': (mad_median, False)
    }

    def __init__(self, n_classes=None, max_depth=np.inf, min_samples_split=2, 
                 criterion_name='gini', debug=False):

        assert criterion_name in self.all_criterions.keys(), 'Criterion name must be on of the following: {}'.format(self.all_criterions.keys())
        
        self.n_classes = n_classes
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion_name = criterion_name

        self.depth = 0
        self.root = None # Use the Node class to initialize it later
        self.debug = debug

        
    def make_split(self, feature_index, threshold, X_subset, y_subset):
        """
        Makes split of the provided data subset and target values using provided feature and threshold
        
        Parameters
        ----------
        feature_index : int
            Index of feature to make split with

        threshold : float
            Threshold value to perform split

        X_subset : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the selected subset

        y_subset : np.array of type float with shape (n_objects, n_classes) in classification 
                   (n_objects, 1) in regression 
            One-hot representation of class labels for corresponding subset
        
        Returns
        -------
        (X_left, y_left) : tuple of np.arrays of same type as input X_subset and y_subset
            Part of the providev subset where selected feature x^j < threshold
        (X_right, y_right) : tuple of np.arrays of same type as input X_subset and y_subset
            Part of the providev subset where selected feature x^j >= threshold
        """

        right_ind = X_subset[:, feature_index] >= threshold
        left_ind = ~right_ind
        
        X_left = X_subset[left_ind, :]
        y_left = y_subset[left_ind, :]
        
        X_right = X_subset[right_ind, :]
        y_right = y_subset[right_ind, :]
        
        return (X_left, y_left), (X_right, y_right)
    
    def make_split_only_y(self, feature_index, threshold, X_subset, y_subset):
        """
        Split only target values into two subsets with specified feature and threshold
        
        Parameters
        ----------
        feature_index : int
            Index of feature to make split with

        threshold : float
            Threshold value to perform split

        X_subset : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the selected subset

        y_subset : np.array of type float with shape (n_objects, n_classes) in classification 
                   (n_objects, 1) in regression 
            One-hot representation of class labels for corresponding subset
        
        Returns
        -------
        y_left : np.array of type float with shape (n_objects_left, n_classes) in classification 
                   (n_objects, 1) in regression 
            Part of the provided subset where selected feature x^j < threshold

        y_right : np.array of type float with shape (n_objects_right, n_classes) in classification 
                   (n_objects, 1) in regression 
            Part of the provided subset where selected feature x^j >= threshold
        """

        (X_left, y_left), (X_right, y_right) = self.make_split(feature_index, \
                                                                threshold, \
                                                                X_subset, \
                                                                y_subset)
        
        return y_left, y_right

    def choose_best_split(self, X_subset, y_subset):
        """
        Greedily select the best feature and best threshold w.r.t. selected criterion
        
        Parameters
        ----------
        X_subset : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the selected subset

        y_subset : np.array of type float with shape (n_objects, n_classes) in classification 
                   (n_objects, 1) in regression 
            One-hot representation of class labels or target values for corresponding subset
        
        Returns
        -------
        feature_index : int
            Index of feature to make split with

        threshold : float
            Threshold value to perform split

        """
        
        n_obj, n_feat = X_subset.shape
        n_classes = self.n_classes
        criteria, is_classification = self.all_criterions[self.criterion_name]
        best_feat = self.DEF_FEAT_IND
        best_threshold = self.DEF_THR
        max_info_delta = 0
        
        node_info = criteria(y_subset)
        
        if n_obj <= self.min_samples_split:
            return best_feat, best_threshold
        
        for feat_ind in range(n_feat):
            feat_col = X_subset[:, feat_ind]
            thresholds = np.unique(feat_col)
        
            for thr in thresholds:
                y_left, y_right = self.make_split_only_y(feat_ind, thr, \
                                                        X_subset, y_subset)
                
                if y_left.shape[0] == 0 or y_right.shape[0] == 0:
                    continue
                
                left_info = y_left.shape[0] / n_obj * criteria(y_left)
                right_info = y_right.shape[0] / n_obj * criteria(y_right)
                info_delta = node_info - left_info - right_info
                
                if info_delta > max_info_delta:
                    max_info_delta = info_delta
                    best_feat = feat_ind
                    best_threshold = thr
                
                
        return best_feat, best_threshold
    
    def make_tree(self, X_subset, y_subset):
        """
        Recursively builds the tree
        
        Parameters
        ----------
        X_subset : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the selected subset

        y_subset : np.array of type float with shape (n_objects, n_classes) in classification 
                   (n_objects, 1) in regression 
            One-hot representation of class labels or target values for corresponding subset
        
        Returns
        -------
        root_node : Node class instance
            Node of the root of the fitted tree
        """
        
        feat_ind, thr = self.choose_best_split(X_subset, y_subset)
        
        if self.depth >= self.max_depth or \
                        (feat_ind == self.DEF_FEAT_IND and thr == self.DEF_THR):
            node = Node(self.DEF_FEAT_IND, self.DEF_THR)
            if self.classification:
                node.value = get_freq_elem(one_hot_decode(y_subset))
                node.proba = get_classes_proba(y_subset)
            else:
                node.value = y_subset.mean() if self.criterion_name == 'variance' \
                                            else np.median(y_subset)
            return node
                
        (X_left, y_left), (X_right, y_right) = \
                                self.make_split(feat_ind, thr, X_subset, y_subset)
        assert(y_left.size > 0 and y_right.size > 0)
        
        new_node = Node(feat_ind, thr)
        self.depth += 1
        new_node.left_child = self.make_tree(X_left, y_left)
        new_node.right_child = self.make_tree(X_right, y_right)
        self.depth -= 1
        
        return new_node
        
    def fit(self, X, y):
        """
        Fit the model from scratch using the provided data
        
        Parameters
        ----------
        X : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the data to train on

        y : np.array of type int with shape (n_objects, 1) in classification 
                   of type float with shape (n_objects, 1) in regression 
            Column vector of class labels in classification or target values in regression
        
        """
        assert len(y.shape) == 2 and len(y) == len(X), 'Wrong y shape'
        self.criterion, self.classification = self.all_criterions[self.criterion_name]
        if self.classification:
            if self.n_classes is None:
                self.n_classes = len(np.unique(y))
            y = one_hot_encode(self.n_classes, y)

        self.root = self.make_tree(X, y)
        
    def __get_obj_prediction(self, obj, get_proba=False):
        node = self.root

        while (node.left_child != None) or (node.right_child != None):
            if obj[node.feature_index] >= node.value:
                node = node.right_child
            else:
                node = node.left_child

        return node.proba if get_proba else node.value
    
    def predict(self, X):
        """
        Predict the target value or class label  the model from scratch using the provided data
        
        Parameters
        ----------
        X : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the data the predictions should be provided for

        Returns
        -------
        y_predicted : np.array of type int with shape (n_objects, 1) in classification 
                   (n_objects, 1) in regression 
            Column vector of class labels in classification or target values in regression
        
        """
        
        y_predicted = list(map(self.__get_obj_prediction, X))
        
        return np.array(y_predicted).reshape(-1, 1)
        
    def predict_proba(self, X):
        """
        Only for classification
        Predict the class probabilities using the provided data
        
        Parameters
        ----------
        X : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the data the predictions should be provided for

        Returns
        -------
        y_predicted_probs : np.array of type float with shape (n_objects, n_classes)
            Probabilities of each class for the provided objects
        
        """
        assert self.classification, 'Available only for classification problem'

        y_predicted_probs = list(map(lambda obj: self.__get_obj_prediction(obj, True), X))
        
        return np.array(y_predicted_probs)
