import numpy as np # use numpy's random number generation
from collections import defaultdict  

from mysklearn import myutils

def train_test_split(X, y, test_size=0.33, random_state=None, shuffle=True):
    """Split dataset into train and test sets based on a test set size.

    Args:
        X(list of list of obj): The list of samples
            The shape of X is (n_samples, n_features)
        y(list of obj): The target y values (parallel to X)
            The shape of y is n_samples
        test_size(float or int): float for proportion of dataset to be in test set (e.g. 0.33 for a 2:1 split)
            or int for absolute number of instances to be in test set (e.g. 5 for 5 instances in test set)
        random_state(int): integer used for seeding a random number generator for reproducible results
            Use random_state to seed your random number generator
                you can use the math module or use numpy for your generator
                choose one and consistently use that generator throughout your code
        shuffle(bool): whether or not to randomize the order of the instances before splitting
            Shuffle the rows in X and y before splitting and be sure to maintain the parallel order of X and y!!

    Returns:
        X_train(list of list of obj): The list of training samples
        X_test(list of list of obj): The list of testing samples
        y_train(list of obj): The list of target y values for training (parallel to X_train)
        y_test(list of obj): The list of target y values for testing (parallel to X_test)

    Note:
        Loosely based on sklearn's train_test_split():
            https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
    """
    if random_state == None:
        random_state = 0
    
    if shuffle:
        myutils.randomize_in_place(X,y)
    
    if isinstance(test_size, float):
        starting_test_index = int(test_size * len(X)) + 1
        X_train = X[: len(X) - starting_test_index]
        X_test = X[starting_test_index + 1:]
        y_train = y[:len(X) - starting_test_index]
        y_test = y[starting_test_index + 1:]
    if isinstance(test_size, int):
        starting_test_index = (len(X) - test_size)
        X_train = X[:starting_test_index]
        X_test = X[starting_test_index:]
        y_train = y[:starting_test_index]
        y_test = y[starting_test_index:]

    return X_train, X_test, y_train, y_test



def kfold_split(X, n_splits=5, random_state=None, shuffle=False):
    """Split dataset into cross validation folds.

    Args:
        X(list of list of obj): The list of samples
            The shape of X is (n_samples, n_features)
        n_splits(int): Number of folds.
        random_state(int): integer used for seeding a random number generator for reproducible results
        shuffle(bool): whether or not to randomize the order of the instances before creating folds

    Returns:
        folds(list of 2-item tuples): The list of folds where each fold is defined as a 2-item tuple
            The first item in the tuple is the list of training set indices for the fold
            The second item in the tuple is the list of testing set indices for the fold

    Notes:
        The first n_samples % n_splits folds have size n_samples // n_splits + 1,
            other folds have size n_samples // n_splits, where n_samples is the number of samples
            (e.g. 11 samples and 4 splits, the sizes of the 4 folds are 3, 3, 3, 2 samples)
        Loosely based on sklearn's KFold split():
            https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html
    """

    n = len(X)
    indices = np.arange(n)
    
    if random_state is not None:
        np.random.seed(random_state)
    
    if shuffle:
        np.random.shuffle(indices)
    
    fold_sizes = [n // n_splits] * n_splits
    for i in range(n % n_splits):
        fold_sizes[i] += 1
    
    current = 0
    folds = []
    for fold_size in fold_sizes:
        test_indices = indices[current:current + fold_size].tolist()
        train_indices = np.delete(indices, slice(current, current + fold_size)).tolist()
        folds.append((train_indices, test_indices))
        current += fold_size
    
    return folds

# BONUS function
def stratified_kfold_split(X, y, n_splits=5, random_state=None, shuffle=False):
    """Split dataset into stratified cross validation folds.

    Args:
        X(list of list of obj): The list of instances (samples).
            The shape of X is (n_samples, n_features)
        y(list of obj): The target y values (parallel to X).
            The shape of y is n_samples
        n_splits(int): Number of folds.
        random_state(int): integer used for seeding a random number generator for reproducible results
        shuffle(bool): whether or not to randomize the order of the instances before creating folds

    Returns:
        folds(list of 2-item tuples): The list of folds where each fold is defined as a 2-item tuple
            The first item in the tuple is the list of training set indices for the fold
            The second item in the tuple is the list of testing set indices for the fold

    Notes:
        Loosely based on sklearn's StratifiedKFold split():
            https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html#sklearn.model_selection.StratifiedKFold
    """
    X = np.array(X)
    y = np.array(y)

    if random_state is not None:
        np.random.seed(random_state)

    if shuffle:
        indices = np.arange(len(y))
        np.random.shuffle(indices)
        X = X[indices]
        y = y[indices]

    # Get the unique classes and their corresponding indices
    unique_classes, class_indices = np.unique(y, return_inverse=True)
    class_indices_dict = {cls: np.where(class_indices == i)[0] for i, cls in enumerate(unique_classes)}

    folds = []

    # Perform stratified splitting
    for cls in unique_classes:
        indices = class_indices_dict[cls]
        n_samples = len(indices)

        fold_sizes = np.full(n_splits, n_samples // n_splits, dtype=int)
        fold_sizes[:n_samples % n_splits] += 1  # distribute the remainder

        current = 0
        for fold_index in range(n_splits):
            start, stop = current, current + fold_sizes[fold_index]
            test_indices = indices[start:stop]  # Testing set for this fold
            
            # Prepare training indices (all other indices)
            train_indices = np.concatenate(
                [indices[:start], indices[stop:]]
            )  # Concatenate all other class indices except current test_indices
            
            folds.append((list(train_indices), list(test_indices)))
            current = stop

    return folds



def bootstrap_sample(X, y=None, n_samples=None, random_state=None):
    """Split dataset into bootstrapped training set and out of bag test set.

    Args:
        X(list of list of obj): The list of samples
        y(list of obj): The target y values (parallel to X)
            Default is None (in this case, the calling code only wants to sample X)
        n_samples(int): Number of samples to generate. If left to None (default) this is automatically
            set to the first dimension of X.
        random_state(int): integer used for seeding a random number generator for reproducible results

    Returns:
        X_sample(list of list of obj): The list of samples
        X_out_of_bag(list of list of obj): The list of "out of bag" samples (e.g. left-over samples)
        y_sample(list of obj): The list of target y values sampled (parallel to X_sample)
            None if y is None
        y_out_of_bag(list of obj): The list of target y values "out of bag" (parallel to X_out_of_bag)
            None if y is None
    Notes:
        Loosely based on sklearn's resample():
            https://scikit-learn.org/stable/modules/generated/sklearn.utils.resample.html
        Sample indexes of X with replacement, then build X_sample and X_out_of_bag
            as lists of instances using sampled indexes (use same indexes to build
            y_sample and y_out_of_bag)
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    n_samples = n_samples or len(X)
    indices = np.random.randint(0, len(X), size=n_samples)
    
    X_sample = [X[i] for i in indices]
    X_out_of_bag = [X[i] for i in range(len(X)) if i not in indices]
    
    if y is not None:
        y_sample = [y[i] for i in indices]
        y_out_of_bag = [y[i] for i in range(len(y)) if i not in indices]
    else:
        y_sample = None
        y_out_of_bag = None
    
    return X_sample, X_out_of_bag, y_sample, y_out_of_bag

def confusion_matrix(y_true, y_pred, labels):
    """Compute confusion matrix to evaluate the accuracy of a classification.

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        labels(list of str): The list of all possible target y labels used to index the matrix

    Returns:
        matrix(list of list of int): Confusion matrix whose i-th row and j-th column entry
            indicates the number of samples with true label being i-th class
            and predicted label being j-th class

    Notes:
        Loosely based on sklearn's confusion_matrix():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
    """
    label_to_index = {label: i for i, label in enumerate(labels)}
    matrix = [[0] * len(labels) for _ in range(len(labels))]
    
    for true, pred in zip(y_true, y_pred):
        true_index = label_to_index[true]
        pred_index = label_to_index[pred]
        matrix[true_index][pred_index] += 1
    
    return matrix

def accuracy_score(y_true, y_pred, normalize=True):
    """Compute the classification prediction accuracy score.

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        normalize(bool): If False, return the number of correctly classified samples.
            Otherwise, return the fraction of correctly classified samples.

    Returns:
        score(float): If normalize == True, return the fraction of correctly classified samples (float),
            else returns the number of correctly classified samples (int).

    Notes:
        Loosely based on sklearn's accuracy_score():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html#sklearn.metrics.accuracy_score
    """
    correct_count = sum(1 for true, pred in zip(y_true, y_pred) if true == pred)
    if normalize:
        return correct_count / len(y_true)
    else:
        return correct_count