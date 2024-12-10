# TODO: your reusable general-purpose functions here
import csv
import numpy as np
import math
import mysklearn.myevaluation as myevaluation

def read_csv(file_path):
    """Reads a CSV file and returns a list of dictionaries."""
    data = []
    
    with open(file_path, mode='r') as file:
        reader = csv.DictReader(file)
        
        for row in reader:
            data.append(row)
    
    return data

def label_encoder(column):
    """Encodes a categorical column as integers."""
    # Ensure the column is a numpy array and flattened
    column = np.array(column).flatten()
    
    # Create a dictionary that maps unique categories to integer labels
    unique_values = list(set(column))  # Extract unique values
    value_to_int = {value: idx for idx, value in enumerate(unique_values)}  # Map values to integers
    
    # Encode the column by mapping each value to its corresponding integer
    encoded_column = np.array([value_to_int[value] for value in column])
    
    return encoded_column  # Return as a numpy array

def compute_euclidean_distance(v1, v2):
    
    isitString = False
    for i in range(len(v1)):
        if isinstance(v1[i], str) or isinstance(v2[i], str):
            isitString = True
    
    dist = 0
    if not isitString:
        dist = math.dist(v1, v2)
    else:
        if v1 == v2:
            dist = 0
        else:
            dist = 1
    return dist

def evaluate_classifier(y_true, y_pred, classifier_name, step, labels, pos_label):
    """Evaluate the performance of a classifier using various metrics.

    Args:
        y_true (array-like): True labels for the test set.
        y_pred (array-like): Predicted labels for the test set.
        classifier_name (str): The name of the classifier (e.g., 'Naive Bayes').
        step (int): The current step number in the cross-validation process.
        labels (list): List of possible label values (e.g., [0, 1]).
        pos_label (int/str): The positive label for binary classification (e.g., 1).
    """
    print(f"\n{'='*38}")
    print(f"STEP {step}: {classifier_name}")
    print("k=10 Stratified K-Fold Cross Validation")
    print(f"{'='*38}")

    # Calculate performance metrics
    accuracy = myevaluation.accuracy_score(y_true, y_pred)
    error_rate = 1 - accuracy
    precision = myevaluation.binary_precision_score(y_true, y_pred, labels, pos_label)
    recall = myevaluation.binary_recall_score(y_true, y_pred, labels, pos_label)
    f1_score = myevaluation.binary_f1_score(y_true, y_pred, labels, pos_label)
    confusion_mat = myevaluation.confusion_matrix(y_true, y_pred, labels)

    # Print results
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Error Rate: {error_rate:.4f}\n")
    print(f"Precision Score: {precision:.4f}")
    print(f"Recall Score: {recall:.4f}")
    print(f"F1 Score: {f1_score:.4f}\n")
    
    # Display confusion matrix
    print("Confusion Matrix:")
    print(confusion_mat)

def get_column(table, col_index):
    col = []
    for i, row in enumerate(table):
        try:
            col.append(row[col_index])
        except:
            print("error")
            pass
    return col

def compute_random_subset(values, num_values):
    values_copy = values.copy()
    np.random.shuffle(values_copy) # inplace shuffle
    return values_copy[:num_values]

def get_majority_vote(predictions):
    instance_predictions, frequencies = get_list_frequencies(predictions)
    most_votes = instance_predictions[0]
    num_votes = frequencies[0]
    for i in range(len(instance_predictions)):
        if frequencies[i] > num_votes:
            num_votes = frequencies[i]
            most_votes = instance_predictions[i]
    return most_votes

def get_list_frequencies(list):
    values = [] 
    counts = [] 
    for value in list:
        if value not in values:
            values.append(value)
            counts.append(1)
        else:
            counts[values.index(value)] += 1 
    return values, counts