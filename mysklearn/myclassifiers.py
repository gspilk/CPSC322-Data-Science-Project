
# TODO: copy your myclassifiers.py solution from PA4-5 here
"""
Brandon Poblette
CPSC 322-01, Fall 2024
Programming Assignment #4
12/12/2024
I attempted the bomus

Description: Create several ml classifiers
"""

import numpy as np
import random
from collections import Counter
from mysklearn import myutils
from mysklearn.mysimplelinearregressor import MySimpleLinearRegressor
from collections import Counter
import statistics
import operator
from mysklearn import myevaluation

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
    """Represents a simple k nearest neighbors classifier.
    Attributes:
        n_neighbors(int): number of k neighbors
        X_train(list of list of numeric vals): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
        y_train(list of obj): The target y values (parallel to X_train).
            The shape of y_train is n_samples
    Notes:
        Loosely based on sklearn's KNeighborsClassifier:
            https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
        Terminology: instance = sample = row and attribute = feature = column
        Assumes data has been properly normalized before use.
    """
    def __init__(self, n_neighbors=3):
        """Initializer for MyKNeighborsClassifier.
        Args:
            n_neighbors(int): number of k neighbors
        """
        self.n_neighbors = n_neighbors
        self.X_train = None
        self.y_train = None

    def fit(self, X_train, y_train):
        """Fits a kNN classifier to X_train and y_train.
        Args:
            X_train(list of list of numeric vals): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples
        Notes:
            Since kNN is a lazy learning algorithm, this method just stores X_train and y_train
        """
        self.X_train = X_train
        self.y_train = y_train

    def kneighbors(self, X_test):
        """Determines the k closes neighbors of each test instance.
        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)
        Returns:
            distances(list of list of float): 2D list of k nearest neighbor distances
                for each instance in X_test
            neighbor_indices(list of list of int): 2D list of k nearest neighbor
                indices in X_train (parallel to distances)
        """
        distances = []
        neighbor_indices = []
        neighbor_index = 0
        for neighbor_index, train_instance in enumerate(self.X_train):
            distance = myutils.compute_euclidean_distance(train_instance, X_test[0])
            distances.append([distance])
            neighbor_indices.append([neighbor_index])
        return distances, neighbor_indices

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.
        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)
        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        distances, neighbor_indices = self.kneighbors(X_test)
        distance_and_indexes = {}
        for i in range(len(distances)):
            distance_and_indexes[neighbor_indices[i][0]] = distances[i][0]
        sorted_distances = sorted(distance_and_indexes.items(), key=operator.itemgetter(-1))
        k_nearest = sorted_distances[:self.n_neighbors]
        y_predicted_k = []
        for index in k_nearest:
            y_predicted_k.append(self.y_train[index[0]])
            
        most_common = statistics.mode(y_predicted_k)
        return most_common



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
        self.priors = {}
        self.posteriors = {}

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
        #count the instances of each class 
        feature_counts = {}
        class_counts = Counter(y_train)
        self.priors = {label: count / len(y_train) for label, count in class_counts.items()}

        #count feature values per class
        feature_counts = {label: [{} for _ in range(len(X_train[0]))] for label in class_counts}


        # Count feature values per class
        for features, label in zip(X_train, y_train):
            for idx, feature in enumerate(features):
                feature_counts[label][idx][feature] = feature_counts[label][idx].get(feature, 0) + 1

        # Calculate probabilities
        self.posteriors = {
            cls: [
                {feature: count / sum(counts[idx].values()) for feature, count in counts[idx].items()}
                for idx in range(len(X_train[0]))
            ]
            for cls, counts in feature_counts.items()
        }

            


    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of obj): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        y_predicted = []

        # Iterate through each test instance
        for features in X_test:
            # Compute log probabilities for all classes
            log_probs = {
                cls: np.log(self.priors[cls]) + sum(
                    np.log(class_posts[idx].get(feature, 1e-6))
                    for idx, feature in enumerate(features)
                )
                for cls, class_posts in self.posteriors.items()
            }

            # Predict the class with the maximum log probability
            y_predicted.append(max(log_probs, key=log_probs.get))

        return y_predicted

class MyDecisionTreeClassifier:
    """Represents a decision tree classifier.

    Attributes:
        X_train (list of list of obj): The list of training instances (samples).
        y_train (list of obj): The target y values (parallel to X_train).
        tree (nested list): The extracted tree model.

    Notes:
        Loosely based on sklearn's DecisionTreeClassifier.
    """
    
    def __init__(self):
        """Initializes MyDecisionTreeClassifier."""
        self.X_train = None
        self.y_train = None
        self.tree = None

    def fit(self, X_train, y_train):
        """Fits a decision tree classifier to X_train and y_train using the TDIDT
        (top down induction of decision tree) algorithm.

        Args:
            X_train (list of list of obj): The list of training instances (samples).
            y_train (list of obj): The target y values (parallel to X_train).
        """
        self.X_train = X_train
        self.y_train = y_train
        available_attributes = list(range(len(X_train[0])))
        self.tree = self._tdidt(X_train, y_train, available_attributes)

    def _tdidt(self, X, y, available_attributes, parent_sample_count=None):
        """Builds the decision tree recursively."""
        if parent_sample_count is None:
            parent_sample_count = len(y)
        
        # Base case: if all samples have the same label
        if len(set(y)) == 1:
            return ["Leaf", y[0], len(y), parent_sample_count]

        # Base case: if no more attributes to split on
        if not available_attributes:
            majority_class = max(set(y), key=y.count)
            return ["Leaf", majority_class, len(y), parent_sample_count]

        best_attr = self._choose_best_attribute(X, y, available_attributes)
        available_attributes = [attr for attr in available_attributes if attr != best_attr]
        tree = ["Attribute", f"att{best_attr}"]

        unique_values = sorted(set(row[best_attr] for row in X))
        for value in unique_values:
            X_sub, y_sub = self._split_dataset(X, y, best_attr, value)
            if not y_sub:
                majority_class = max(set(y), key=y.count)
                tree.append(["Value", value, ["Leaf", majority_class, len(y_sub), len(y)]])
            else:
                subtree = self._tdidt(X_sub, y_sub, available_attributes.copy(), len(y))
                tree.append(["Value", value, subtree])

        return tree

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test (list of list of obj): The list of testing samples.

        Returns:
            y_predicted (list of obj): The predicted target y values.
        """
        return [self._traverse_tree(self.tree, instance) or "A" for instance in X_test]

    def _traverse_tree(self, node, instance):
        """Traverses the decision tree and makes a prediction."""
        if node[0] == "Leaf":
            return node[1]
        
        attr_index = int(node[1][3:])
        for branch in node[2:]:
            if branch[0] == "Value" and branch[1] == instance[attr_index]:
                return self._traverse_tree(branch[2], instance)
        return None

    def _choose_best_attribute(self, X, y, attributes):
        """Chooses the best attribute to split on using information gain."""
        base_entropy = self._entropy(y)
        best_attr, best_gain = None, -1

        for attr in attributes:
            sub_entropies = [
                (len(y_sub) / len(y)) * self._entropy(y_sub)
                for value in set(row[attr] for row in X)
                if (X_sub := [row for row in X if row[attr] == value]) and (y_sub := [y[i] for i in range(len(y)) if X[i][attr] == value])
            ]
            info_gain = base_entropy - sum(sub_entropies)
            if info_gain > best_gain:
                best_gain, best_attr = info_gain, attr

        return best_attr

    def _split_dataset(self, X, y, attr, value):
        """Splits the dataset based on an attribute and value."""
        X_sub = [row for row in X if row[attr] == value]
        y_sub = [y[i] for i in range(len(y)) if X[i][attr] == value]
        return X_sub, y_sub

    def _entropy(self, y):
        """Calculates the entropy of a list of labels."""
        from math import log2
        total = len(y)
        return -sum((y.count(label) / total) * log2(y.count(label) / total) for label in set(y))

    def print_decision_rules(self, attribute_names=None, class_name="class"):
        """Prints the decision rules from the tree."""
        rules = []
        self._traverse_tree_for_rules(self.tree, rules, [], attribute_names, class_name)
        print("\n".join(" AND ".join(rule[:-1]) + " THEN " + rule[-1] for rule in rules))

    def _traverse_tree_for_rules(self, node, rules, rule, attribute_names, class_name):
        """Traverses the tree and collects decision rules."""
        if node[0] == "Leaf":
            rule.append(f"{class_name} = {node[1]}")
            rules.append(rule)
            return

        attr_index = int(node[1][3:])
        for branch in node[2:]:
            new_rule = rule.copy()
            attr_name = attribute_names[attr_index] if attribute_names else f"att{attr_index}"
            new_rule.append(f"IF {attr_name} == {branch[1]}")
            self._traverse_tree_for_rules(branch[2], rules, new_rule, attribute_names, class_name)


    # BONUS method
    def visualize_tree(self, dot_fname, pdf_fname, attribute_names=None):
        """BONUS: Visualizes a tree via the open source Graphviz graph visualization package and
        its DOT graph language (produces .dot and .pdf files).

        Args:
            dot_fname(str): The name of the .dot output file.
            pdf_fname(str): The name of the .pdf output file generated from the .dot file.
            attribute_names(list of str or None): A list of attribute names to use in the decision rules
                (None if a list is not provided and the default attribute names based on indexes
                (e.g. "att0", "att1", ...) should be used).

        Notes:
            Graphviz: https://graphviz.org/
            DOT language: https://graphviz.org/doc/info/lang.html
            You will need to install graphviz in the Docker container as shown in class to complete this method.
        """
        pass # TODO: (BONUS) fix this


class MyRandomForestClassifier():
    """
    Generate a random stratified test set consisting of one third of the original data set, with the remaining two thirds of the instances forming the "remainder set".
    Generate N "random" decision trees using bootstrapping (giving a training and validation set) over the remainder set. At each node, build your decision trees by randomly selecting F of the remaining attributes as candidates to partition on. This is the standard random forest approach discussed in class. Note that to build your decision trees you should still use entropy; however, you are selecting from only a (randomly chosen) subset of the available attributes.
    Select the M most accurate of the N decision trees using the corresponding validation sets.
    Use simple majority voting to predict classes using the M decision trees over the test set.

    """
    def __init__(self, N, M, F):
        self.N = N  # Number of trees
        self.M = M  # Number of trees to keep
        self.F = F  # Number of features to randomly select for each tree
        self.trees = []  # List to store the decision trees
        self.attribute_domain = {}  # Domain for each attribute in the dataset
        self.header = []  # Attribute names

    def fit(self, X, y):
        """Fits the Random Forest model using bootstrapping and decision trees.
        
        Args:
            X (list of list): Training data instances (samples).
            y (list): Target values corresponding to the training data.
        """
        # Define attribute names (att0, att1, ...)
        self.header = [f"att{i}" for i in range(len(X[0]))]

        # Construct the attribute domain (unique values for each attribute)
        self.attribute_domain = {
            header: list(np.unique(myutils.get_column(X, i)))
            for i, header in enumerate(self.header)
        }

        # Generate N decision trees
        N_trees = []
        for _ in range(self.N):
            X_train, X_test, y_train, y_test = myevaluation.bootstrap_sample(X, y)

            # Create and train a decision tree using a random subset of features
            available_attributes = myutils.compute_random_subset(self.header, self.F)
            curr_tree = MyDecisionTreeClassifier()
            curr_tree.fit(X_train, y_train)

            # Evaluate the tree on the validation set
            curr_predictions = curr_tree.predict(X_test)
            accuracy = myevaluation.accuracy_score(y_test, curr_predictions)
            N_trees.append((accuracy, curr_tree))

        # Sort the trees by accuracy and keep the top M
        N_trees.sort(key=lambda x: x[0], reverse=True)
        self.trees = [tree for _, tree in N_trees[:self.M]]

    def predict(self, X_test):
        """Makes predictions for the given test instances.

        Args:
            X_test (list of list): Testing data instances.

        Returns:
            list: Predicted target values for the test set.
        """
        # Collect predictions from all trees
        all_predictions = [tree.predict(X_test) for tree in self.trees]

        # Use majority voting to make final predictions
        y_predicted = []
        for i in range(len(all_predictions[0])):
            instance_votes = myutils.get_column(all_predictions, i)
            y_predicted.append(myutils.get_majority_vote(instance_votes))

        return y_predicted


