import numpy as np


class KNNClassifier:
    """
    A k-Nearest Neighbors (k-NN) classifier for supervised learning tasks.

    This implementation calculates distances between data points to classify
    test data based on the majority label of its nearest neighbors in the training set.

    Attributes:
        num_neighbors (int): The number of nearest neighbors to consider during classification.
        train_features (numpy.ndarray): Feature matrix for the training data.
        class_labels (numpy.ndarray): Labels corresponding to the training data.
        test_features (numpy.ndarray): Feature matrix for the test data.
        norm_dict (dict): Dictionary of the distance functions corresponding to the distance metrics.

    Methods:
        __init__():
            Initializes the k-NN classifier with optional input parameters.

        knn_classifier(train_features, class_labels):
            Returns the label predictions for test features given the train features, class labels, number of neighbors,
             and test features. Also returns the probabilities if return_probabilities is True

        get_training_predictions(test_features): Returns the label predictions for the training data points based on
        their neighbors
    """
    norm_dict = {
        "L2": lambda x, y: np.linalg.norm(x - y, axis=-1),
        "L1": lambda x, y: np.sum(np.abs(x - y, axis=-1)),
    }

    def __init__(self, **kwargs):
        """
        Initialize the KNNClassifier with optional attributes.

        Example:
            knn = KNNClassifier(distance_metric=L2, num_neighbors=3)
        """
        self.distance_metric = "L2"
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.set_distance_metric(self.distance_metric)

    def set_distance_metric(self, distance_metric):
        if distance_metric in self.norm_dict.keys():
            self.distance_metric = distance_metric
            self.distance_func = self.norm_dict[self.distance_metric]
        else:
            raise ValueError(f"Unsupported distance metric: {distance_metric}")

    def set_train_data(self, train_features, class_labels):
        if len(train_features.shape) != 2:
            raise ValueError("train_features should be a 2D array.")
        if len(class_labels.shape) != 1:
            raise ValueError("train_features should be a 1D array.")
        self.train_features = train_features
        self.class_labels = class_labels
        self.num_classes = len(np.unique(self.class_labels))
        self.num_observations = self.train_features.shape[0]
        self.num_variables = self.train_features.shape[1]

    def set_test_data(self, test_features):
        if len(test_features.shape) != 2:
            raise ValueError("test_features should be a 2D array.")
        self.test_features = test_features
        self.num_testpoints = self.test_features.shape[0]

    def set_num_neighbors(self, num_neighbors):
        self.num_neighbors = num_neighbors

    def _get_neighbors_labels(self, train_data=False):
        """
        Internal function to compute distances, decide their neighbors and retrieve the labels of the neighbors for
        the inferred feature data.

        Parameters:
            train_data (bool): Whether to predict labels for train or test features

        Returns:
            numpy.ndarray: The labels of the neighbors for each point, shape (num_features, num_neighbors)
        """
        if train_data:
            distances = self.distance_func(self.train_features[None, :, :], self.train_features[:, None, :])
            neighbor_indices = np.argsort(distances, axis=-1)[:, 1:(self.num_neighbors + 1)]
        else:
            distances = self.distance_func(self.train_features[None, :, :], self.test_features[:, None, :])
            neighbor_indices = np.argsort(distances, axis=-1)[:, :self.num_neighbors]
        return self.class_labels[neighbor_indices]


    def _predict(self, train_data=False):
        """
        Internal function to predict the labels of the train or test features based on their neighbors among the
        train features.

        Parameters:
            train_data (bool): Whether to predict labels for train or test features

        Returns:
            numpy.ndarray: Predicted class labels.
            list: Probabilities. It is computed by the distribution of labels in its neighbors for each point.
        """
        neighbor_labels = self._get_neighbors_labels(train_data)
        predictions = []
        predictions_proba = []

        for row in neighbor_labels:
            labels, counts = np.unique(row, return_counts=True)
            predictions.append(labels[np.argmax(counts)])
            predictions_proba.append({label: count / self.num_neighbors for label, count in zip(labels, counts)})

        return np.array(predictions), predictions_proba

    def predict_training_data(self, num_neighbors, return_probabilities=False):
        """
        Predict the labels of the train features based on their distances to the other train features.

        Parameters:
            num_neighbors (bool): Whether to predict labels for train or test features
            return_probabilities (bool): Whether to return probabilities as well

        Returns:
            numpy.ndarray: Predicted class labels.
            list: Probabilities. It is computed by the distribution of labels in its neighbors for each point.
        """
        if not hasattr(self, 'train_features'):
            raise ValueError("Set train data before using 'get_training_predictions'.")
        if not hasattr(self, 'class_labels'):
            raise ValueError("Set train data before using 'get_training_predictions'.")
        if not hasattr(self, 'num_neighbors'):
            raise ValueError("Set number of neighbors before using 'get_training_predictions'.")

        self.set_num_neighbors(num_neighbors)
        train_predictions, train_predictions_proba = self._predict(train_data=True)

        if return_probabilities:
            return train_predictions, train_predictions_proba
        else:
            return train_predictions

    def knn_classifier(self, train_features, class_labels, test_features, num_neighbors, return_probabilities=False):
        """
        Predict the labels for the test features given all the input parameters.

        Parameters:
            train_features (numpy.ndarray): Feature matrix for the train data, shape (num_observations, num_variables).
            class_labels (numpy.ndarray): Labels corresponding to the train data, shape (num_observations,).
            test_features (numpy.ndarray): Feature matrix for the test data, shape (n_test_samples, num_variables)
            num_neighbors (int): The number of nearest neighbors to consider during classification.
            return_probabilities (bool): Whether to return prediction probabilities.

        Returns:
            numpy.ndarray: Predicted class labels.
            list: Probabilities. It is computed for each test point by the label distribution of its neighbors.
        """
        self.set_train_data(train_features, class_labels)
        self.set_test_data(test_features)
        self.set_num_neighbors(num_neighbors)

        test_predictions, test_predictions_proba = self._predict(train_data=False)

        if return_probabilities:
            return test_predictions, test_predictions_proba
        else:
            return test_predictions
