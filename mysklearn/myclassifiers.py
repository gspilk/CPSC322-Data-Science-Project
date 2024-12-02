from mysklearn import myutils

# TODO: copy your myclassifiers.py solution from PA4-5 here
"""
Brandon Poblette
CPSC 322-01, Fall 2024
Programming Assignment #4
10/29/2024
I attempted the bomus

Description: Create several ml classifiers
"""
import numpy as np
import random
from collections import Counter
from mysklearn import myutils
from mysklearn.mysimplelinearregressor import MySimpleLinearRegressor

class MySimpleLinearRegressionClassifier:
    """Represents a simple linear regression classifier that discretizes
        predictions from a simple linear regressor (see MySimpleLinearRegressor).

    Attributes:
        discretizer(function): a function that discretizes a numeric value into
            a string label. The function's signature is func(obj) -> obj
        regressor(MySimpleLinearRegressor): the underlying regression model that
            fits a line to x and y data
    """

    def __init__(self, discretizer, regressor=None):
        """Initializer for MySimpleLinearClassifier.

        Args:
            discretizer(function): a function that discretizes a numeric value into
                a string label. The function's signature is func(obj) -> obj
            regressor(MySimpleLinearRegressor): the underlying regression model that
                fits a line to x and y data (None if to be created in fit())
        """
        self.discretizer = discretizer
        self.regressor = regressor or MySimpleLinearRegressor()

    def fit(self, X_train, y_train):
        """Fits a simple linear regression line to X_train and y_train."""
       
        # Initialize regressor if not already done
        if self.regressor is None:
            self.regressor = MySimpleLinearRegressor()
       
        # Fit the underlying regressor
        self.regressor.fit(X_train, y_train)
       
        # Set the slope and intercept from the regressor for easy access
        self.slope = self.regressor.slope
        self.intercept = self.regressor.intercept

    def predict(self, X_test):
        """Makes predictions for test samples in X_test by applying discretizer
            to the numeric predictions from regressor.

        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        # Get continuous predictions from the regressor
        y_pred_continuous = self.regressor.predict(X_test)

        if self.discretizer is None:    
            return y_pred_continuous
        else:
            # Discretize the continuous predictions
            y_pred_discrete = [self.discretizer(y) for y in y_pred_continuous]
            return y_pred_discrete


class MyKNeighborsClassifier:
    """Represents a simple k nearest neighbors classifier."""

    def __init__(self, n_neighbors=3):
        """Initializer for MyKNeighborsClassifier."""
        self.n_neighbors = n_neighbors
        self.X_train = None
        self.y_train = None

    def fit(self, X_train, y_train):
        """Fits a kNN classifier to X_train and y_train."""
        self.X_train = np.array(X_train)
        self.y_train = np.array(y_train)

    def kneighbors(self, X_test):
        """Determines the k closest neighbors of each test instance."""
        X_test = np.array(X_test)
        distances = []
        neighbor_indices = []

        for test_point in X_test:
            # Calculate the Euclidean distances from the test point to all training points
            dists = np.linalg.norm(self.X_train - test_point, axis=1)
            # Get the indices of the k closest neighbors
            k_indices = np.argsort(dists)[:self.n_neighbors]
            # Store the distances and indices of the k neighbors
            distances.append(dists[k_indices].tolist())
            neighbor_indices.append(k_indices.tolist())

        return distances, neighbor_indices

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        y_predicted = []

        for x_test in X_test:
            # Calculate the Euclidean distance from x_test to all training points
            distances = np.linalg.norm(np.array(self.X_train) - np.array(x_test), axis=1)
            # Get the indices of the k nearest neighbors
            neighbor_indices = np.argsort(distances)[:self.n_neighbors]
            # Get the corresponding labels
            neighbor_labels = [self.y_train[i] for i in neighbor_indices]
            # Compute the average for regression
            y_predicted.append(np.mean(neighbor_labels))

        return y_predicted


class MyDummyClassifier:
    """Represents a "dummy" classifier using the "most_frequent" or "stratified" strategy."""

    def __init__(self, strategy="most_frequent"):
        """Initializer for DummyClassifier.

        Args:
            strategy (str): The strategy to use, either "most_frequent" or "stratified".
        """
        self.most_common_label = None
        self.strategy = strategy
        self.label_distribution = None  # To store label distribution for stratified strategy

    def fit(self, X_train, y_train):
        """Fits a dummy classifier to X_train and y_train."""
        # Count the occurrences of each class label in y_train
        label_counts = Counter(y_train)
        self.label_distribution = label_counts  # Store distribution for stratified strategy

        if self.strategy == "most_frequent":
            # Get the most common class label
            self.most_common_label = label_counts.most_common(1)[0][0]  # Get the most frequent label

    def predict(self, X_test):
        """Makes predictions for test instances in X_test."""
        if self.strategy == "most_frequent":
            # Return a list of the most common label for the length of X_test
            return [self.most_common_label] * len(X_test)
        elif self.strategy == "stratified":
            # Calculate total number of samples
            total_count = sum(self.label_distribution.values())
            # Create a list of labels according to their probabilities
            probabilities = [(label, count / total_count) for label, count in self.label_distribution.items()]
            labels, weights = zip(*probabilities)
            # Return a list of randomly selected labels based on their probabilities
            return random.choices(labels, weights=weights, k=len(X_test))
        else:
            raise ValueError("Invalid strategy. Use 'most_frequent' or 'stratified'.")

class MyNaiveBayesClassifier:
    """Represents a Naive Bayes classifier.

    Attributes:
        priors(YOU CHOOSE THE MOST APPROPRIATE TYPE): The prior probabilities computed for each
            label in the training set.
        posteriors(YOU CHOOSE THE MOST APPROPRIATE TYPE): The posterior probabilities computed for each
            attribute value/label pair in the training set.

    Notes:
        Loosely based on sklearn's Naive Bayes classifiers: https://scikit-learn.org/stable/modules/naive_bayes.html
        You may add additional instance attributes if you would like, just be sure to update this docstring
        Terminology: instance = sample = row and attribute = feature = column
    """
    def __init__(self):
        """Initializer for MyNaiveBayesClassifier.
        """
        self.priors = None
        self.posteriors = None

    def fit(self, X_train, y_train):
        """Fits a Naive Bayes classifier to X_train and y_train.

        Args:
            X_train(list of list of obj): The list of training instances (samples)
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples

        Notes:
            Since Naive Bayes is an eager learning algorithm, this method computes the prior probabilities
                and the posterior probabilities for the training data.
            You are free to choose the most appropriate data structures for storing the priors
                and posteriors.
        """
        pass # TODO: fix this

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of obj): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        return [] # TODO: fix this


    class MyRandomForestClassifier():
    """
    
    """