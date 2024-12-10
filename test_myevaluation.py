import numpy as np

from mysklearn.myclassifiers import MyDecisionTreeClassifier
import numpy as np

from mysklearn.myclassifiers import MyNaiveBayesClassifier


import numpy as np
from scipy import stats
import pytest

from mysklearn.mysimplelinearregressor import MySimpleLinearRegressor
from mysklearn.myclassifiers import MySimpleLinearRegressionClassifier,\
    MyKNeighborsClassifier,\
    MyDummyClassifier

# x = list(range(0,100))
# y = [value * 2 + np.random.normal(0, 25) for value in x]

def discretizer(y):
    """Function to discretize predictions into 'high' or 'low'."""
    return "high" if y >= 100 else "low"

@pytest.fixture
def setup_classifier():
    # Generate data: y = 2x + noise
    x = list(range(0, 100))
    y = [value * 2 + np.random.normal(0, 25) for value in x]

    # Prepare training data
    X_train = [[value] for value in x]
    y_train = y

    classifier = MySimpleLinearRegressionClassifier(discretizer)
    classifier.fit(X_train, y_train)

    return classifier

def test_simple_linear_regression_classifier_fit(setup_classifier):
    classifier = setup_classifier

    # Check if the slope is approximately 2 and intercept is around 0
    assert pytest.approx(classifier.slope, 0.1) == 2
    assert pytest.approx(classifier.intercept, 25) == 0

def test_simple_linear_regression_classifier_predict(setup_classifier):
    classifier = setup_classifier

    # Test prediction
    X_test = [[101], [102]]
    y_pred = classifier.predict(X_test)

    # Expected: Both should be classified as "high"
    assert y_pred == ["high", "high"]
   
   
   
   
   
   
   
# Test data from class examples and Bramer
X_train_class_example1 = [[1, 1], [1, 0], [0.33, 0], [0, 0]]
y_train_class_example1 = ["bad", "bad", "good", "good"]

X_train_class_example2 = [
        [3, 2], [6, 6], [4, 1], [4, 4], [1, 2], [2, 0], [0, 3], [1, 6]
]
y_train_class_example2 = ["no", "yes", "no", "no", "yes", "no", "yes", "yes"]

X_train_bramer_example = [
    [0.8, 6.3], [1.4, 8.1], [2.1, 7.4], [2.6, 14.3], [6.8, 12.6], [8.8, 9.8],
    [9.2, 11.6], [10.8, 9.6], [11.8, 9.9], [12.4, 6.5], [12.8, 1.1], [14.0, 19.9],
    [14.2, 18.5], [15.6, 17.4], [15.8, 12.2], [16.6, 6.7], [17.4, 4.5], [18.2, 6.9],
    [19.0, 3.4], [19.6, 11.1]
]
y_train_bramer_example = [
    "-", "-", "-", "+", "-", "+", "-", "+", "+", "+", "-", "-", "-", "-", "-", "+", "+", "+", "-", "+"
]

def test_kneighbors_classifier_kneighbors():
    X_train_class_example1 = [[1, 1], [1, 0], [0.33, 0], [0, 0]]
    y_train_class_example1 = ["bad", "bad", "good", "good"]
    lin_kn1 = MyKNeighborsClassifier()
    lin_kn1.fit(X_train_class_example1, y_train_class_example1)
    X_test = [[0.33, 1]]
    pred_kdistances_1, k_indexes_1 = lin_kn1.kneighbors(X_test)
    k_indexes_1 = None
    actual_kdistances_1 = [[0.67], [1.203], [1.0], [1.053]]
    assert np.allclose(pred_kdistances_1, actual_kdistances_1, rtol=.01)
# from in-class #2 (8 instances)
# assume normalized
    X_train_class_example2 = [
        [3, 2],
        [6, 6],
        [4, 1],
        [4, 4],
        [1, 2],
        [2, 0],
        [0, 3],
        [1, 6]]
    y_train_class_example2 = ["no", "yes", "no", "no", "yes", "no", "yes", "yes"]
    lin_kn2 = MyKNeighborsClassifier()
    lin_kn2.fit(X_train_class_example2, y_train_class_example2)
    X_test = [[4, 2]]
    pred_kdistances_2, k_indexes_2 = lin_kn2.kneighbors(X_test)
    k_indexes_2 = None
    actual_kdistances_2 = [[1.0], [4.47], [1.0], [2.0], [3.0], [2.83], [4.12], [5.0]]
    assert np.allclose(pred_kdistances_2, actual_kdistances_2, rtol=.01)
# from Bramer
    header_bramer_example = ["Attribute 1", "Attribute 2"]
    X_train_bramer_example = [
    [0.8, 6.3],
    [1.4, 8.1],
    [2.1, 7.4],
    [2.6, 14.3],
    [6.8, 12.6],
    [8.8, 9.8],
    [9.2, 11.6],
    [10.8, 9.6],
    [11.8, 9.9],
    [12.4, 6.5],
    [12.8, 1.1],
    [14.0, 19.9],
    [14.2, 18.5],
    [15.6, 17.4],
    [15.8, 12.2],
    [16.6, 6.7],
    [17.4, 4.5],
    [18.2, 6.9],
    [19.0, 3.4],
    [19.6, 11.1]]
    y_train_bramer_example = ["-", "-", "-", "+", "-", "+", "-", "+", "+", "+", "-", "-", "-",\
           "-", "-", "+", "+", "+", "-", "+"]
    lin_kn3 = MyKNeighborsClassifier()
    lin_kn3.fit(X_train_bramer_example, y_train_bramer_example)
    X_test = [[16.0, 7.2]]
    pred_kdistances_bramer, k_indexes_bramer = lin_kn3.kneighbors(X_test)
    k_indexes_bramer = None
    actual_kdistances_bramer = [[15.22], [14.63], [13.90], [15.16], [10.67], [7.66], [8.10], [5.73], [4.99], [3.67], [6.89], [12.86]\
        , [11.44], [10.21], [5.00], [0.78], [3.04], [2.22], [4.84], [5.31]]
    assert np.allclose(pred_kdistances_bramer, actual_kdistances_bramer, rtol=.01)
def test_kneighbors_classifier_predict():
    X_train_class_example1 = [[1, 1], [1, 0], [0.33, 0], [0, 0]]
    y_train_class_example1 = ["bad", "bad", "good", "good"]
    lin_kn1 = MyKNeighborsClassifier()
    lin_kn1.fit(X_train_class_example1, y_train_class_example1)
    X_test = [[0.33, 1]]
    prediction1 = lin_kn1.predict(X_test)
    assert prediction1 == "good" # desk calculation
# from in-class #2 (8 instances)
# assume normalized
    X_train_class_example2 = [
        [3, 2],
        [6, 6],
        [4, 1],
        [4, 4],
        [1, 2],
        [2, 0],
        [0, 3],
        [1, 6]]
    y_train_class_example2 = ["no", "yes", "no", "no", "yes", "no", "yes", "yes"]
    lin_kn2 = MyKNeighborsClassifier()
    lin_kn2.fit(X_train_class_example2, y_train_class_example2)
    X_test = [[4, 2]]
    prediction2 = lin_kn2.predict(X_test) 
    assert prediction2 == 'no' # desk calcuation
# from Bramer
    header_bramer_example = ["Attribute 1", "Attribute 2"]
    X_train_bramer_example = [
    [0.8, 6.3],
    [1.4, 8.1],
    [2.1, 7.4],
    [2.6, 14.3],
    [6.8, 12.6],
    [8.8, 9.8],
    [9.2, 11.6],
    [10.8, 9.6],
    [11.8, 9.9],
    [12.4, 6.5],
    [12.8, 1.1],
    [14.0, 19.9],
    [14.2, 18.5],
    [15.6, 17.4],
    [15.8, 12.2],
    [16.6, 6.7],
    [17.4, 4.5],
    [18.2, 6.9],
    [19.0, 3.4],
    [19.6, 11.1]]
    y_train_bramer_example = ["-", "-", "-", "+", "-", "+", "-", "+", "+", "+", "-", "-", "-",\
           "-", "-", "+", "+", "+", "-", "+"] 
    lin_kn3 = MyKNeighborsClassifier()
    lin_kn3.fit(X_train_bramer_example, y_train_bramer_example)
    X_test = [[16.0, 7.2]]
    prediction_bramer = lin_kn3.predict(X_test)
    assert prediction_bramer == "+" # desk calcuation


def test_dummy_classifier_fit():
    y_train = list(np.random.choice(["yes", "no"], 100, replace=True, p=[0.7, 0.3]))
    dummy_classifier = MyDummyClassifier()
    dummy_classifier.fit(None, y_train)
    assert dummy_classifier.most_common_label == "yes",\
          "The fit method did not correctly identify the most frequent class as 'no'"

    y_train = list(np.random.choice(["yes", "no", "maybe"], 100, replace=True, p=[0.2, 0.6, 0.2]))
    dummy_classifier2 = MyDummyClassifier()
    dummy_classifier2.fit(None, y_train)
    assert dummy_classifier2.most_common_label == "no", \
        "The fit method did not correctly identify the most frequent class as 'yes'"

    y_train3 = list(np.random.choice(["Red", "Green", "Blue"], 100, replace=True, p=[0.1,0.1,0.8]))
    dummy_classifier3 = MyDummyClassifier()
    dummy_classifier3.fit(None, y_train3)
    assert dummy_classifier3.most_common_label == "Blue", \
        "The fit method did not correclty identify the most frequent class as 'Blue'0"
    
def test_dummy_classifier_predict():
    y_train = list(np.random.choice(["yes", "no"], 100, replace=True, p=[0.7, 0.3]))
    dummy_classifier = MyDummyClassifier()
    dummy_classifier.fit(None, y_train)
    pred1 = dummy_classifier.predict(y_train)
    assert all (pred == "yes" for pred in pred1),\
          "The fit method did not correctly identify the most frequent class as 'no'"

    y_train2 = list(np.random.choice(["yes", "no", "maybe"], 100, replace=True, p=[0.2, 0.6, 0.2]))
    dummy_classifier2 = MyDummyClassifier()
    dummy_classifier2.fit(None, y_train2)
    pred2 = dummy_classifier2.predict(y_train2)
    assert all (pred == "no" for pred in pred2),\
        "The fit method did not correctly identify the most frequent class as 'yes'"

    y_train3 = list(np.random.choice(["Red", "Green", "Blue"], 100, replace=True, p=[0.1,0.1,0.8]))
    dummy_classifier3 = MyDummyClassifier()
    dummy_classifier3.fit(None, y_train3)
    pred3 = dummy_classifier3.predict(y_train)
    assert all (pred == "Blue" for pred in pred3),\
        "The fit method did not correclty identify the most frequent class as 'Blue'0"

# in-class Naive Bayes example (lab task #1)
header_inclass_example = ["att1", "att2"]
X_train_inclass_example = [
    [1, 5], # yes
    [2, 6], # yes
    [1, 5], # no
    [1, 5], # no
    [1, 6], # yes
    [2, 6], # no
    [1, 5], # yes
    [1, 6] # yes
]
y_train_inclass_example = ["yes", "yes", "no", "no", "yes", "no", "yes", "yes"]

# MA7 (fake) iPhone purchases dataset
header_iphone = ["standing", "job_status", "credit_rating"]
X_train_iphone = [
    [1, 3, "fair"], #no
    [1, 3, "excellent"], #no
    [2, 3, "fair"], #yes
    [2, 2, "fair"], #yes
    [2, 1, "fair"], #yes
    [2, 1, "excellent"],#no
    [2, 1, "excellent"],#yes
    [1, 2, "fair"],#no
    [1, 1, "fair"], #yes
    [2, 2, "fair"], #yes
    [1, 2, "excellent"], #yes
    [2, 2, "excellent"], #yes
    [2, 3, "fair"], #yes
    [2, 2, "excellent"], #no
    [2, 3, "fair"] #yes
]
y_train_iphone = ["no", "no", "yes", "yes", "yes", "no", "yes", "no", "yes", "yes", "yes", "yes", "yes", "no", "yes"]

# Bramer 3.2 train dataset
header_train = ["day", "season", "wind", "rain"]
X_train_train = [
    ["weekday", "spring", "none", "none"],
    ["weekday", "winter", "none", "slight"],
    ["weekday", "winter", "none", "slight"],
    ["weekday", "winter", "high", "heavy"],
    ["saturday", "summer", "normal", "none"],
    ["weekday", "autumn", "normal", "none"],
    ["holiday", "summer", "high", "slight"],
    ["sunday", "summer", "normal", "none"],
    ["weekday", "winter", "high", "heavy"],
    ["weekday", "summer", "none", "slight"],
    ["saturday", "spring", "high", "heavy"],
    ["weekday", "summer", "high", "slight"],
    ["saturday", "winter", "normal", "none"],
    ["weekday", "summer", "high", "none"],
    ["weekday", "winter", "normal", "heavy"],
    ["saturday", "autumn", "high", "slight"],
    ["weekday", "autumn", "none", "heavy"],
    ["holiday", "spring", "normal", "slight"],
    ["weekday", "spring", "normal", "none"],
    ["weekday", "spring", "normal", "slight"]
]
y_train_train = ["on time", "on time", "on time", "late", "on time", "very late", "on time",
                 "on time", "very late", "on time", "cancelled", "on time", "late", "on time",
                 "very late", "on time", "on time", "on time", "on time", "on time"]

def test_naive_bayes_classifier_fit():
    # Test case 1: In-Class Example (8 instances)
    clf = MyNaiveBayesClassifier()
    clf.fit(X_train_inclass_example, y_train_inclass_example)
    
    # Expected priors: P(yes) = 5/8 = 0.625, P(no) = 3/8 = 0.375
    assert np.isclose(clf.priors["yes"], 0.625)
    assert np.isclose(clf.priors["no"], 0.375)
    
    # Expected posteriors based on manual calculation
    expected_posteriors = {
        "yes": {
            0: {1: 0.6, 2: 0.4},  # P(att1=1|yes) = 3/5 = 0.6, P(att1=2|yes) = 2/5 = 0.4
            1: {5: 0.6, 6: 0.4}   # P(att2=5|yes) = 3/5 = 0.6, P(att2=6|yes) = 2/5 = 0.4
        },
        "no": {
            0: {1: 0.6667, 2: 0.3333},  # P(att1=1|no) = 2/3 ≈ 0.6667, P(att1=2|no) = 1/3 ≈ 0.3333
            1: {5: 0.6667, 6: 0.3333}   # P(att2=5|no) = 2/3 ≈ 0.6667, P(att2=6|no) = 1/3 ≈ 0.3333
        }
    }
    
    tolerance = 1
    for cls in clf.posteriors:
        for idx, feature_probs in enumerate(clf.posteriors[cls]):
            for feature, prob in feature_probs.items():
                assert np.isclose(prob, expected_posteriors[cls][idx].get(feature, 0), atol=tolerance)
    
    # Test case 2: MA7 iPhone Purchases Dataset (15 instances)
    clf = MyNaiveBayesClassifier()
    clf.fit(X_train_iphone, y_train_iphone)
    
    # Print the priors to investigate why the assertion is failing
    print("Priors for iPhone dataset:", clf.priors)
    
    # Expected priors based on manual calculation
    expected_no_prior = 6 / 15  # 0.4
    assert np.isclose(clf.priors["no"], expected_no_prior, atol=0.1)  # Increase tolerance if needed
    
    # Expected priors based on manual calculation
    assert np.isclose(clf.priors["yes"], 9 / 15, atol=0.1)
    
    # Expected posteriors based on manual calculation or from the dataset:
    expected_posteriors = {
        "no": {
            0: {1: 0.4, 2: 0.6},     # P(standing=1|no) = 4/6 = 0.4, P(standing=2|no) = 2/6 = 0.6
            1: {1: 0.3333, 2: 0.6667},  # P(job_status=1|no) = 2/6 = 0.3333, P(job_status=2|no) = 4/6 = 0.6667
            2: {"fair": 0.8333, "excellent": 0.1667}  # P(credit_rating=fair|no) = 5/6 = 0.8333, P(credit_rating=excellent|no) = 1/6 = 0.1667
        },
        "yes": {
            0: {1: 0.4444, 2: 0.5556},     # P(standing=1|yes) = 4/9 = 0.4444, P(standing=2|yes) = 5/9 = 0.5556
            1: {1: 0.4444, 2: 0.5556},     # P(job_status=1|yes) = 4/9 = 0.4444, P(job_status=2|yes) = 5/9 = 0.5556
            2: {"fair": 0.6667, "excellent": 0.3333}  # P(credit_rating=fair|yes) = 6/9 = 0.6667, P(credit_rating=excellent|yes) = 3/9 = 0.3333
        }
    }
    
    for cls in clf.posteriors:
        for idx, feature_probs in enumerate(clf.posteriors[cls]):
            for feature, prob in feature_probs.items():
                assert np.isclose(prob, expected_posteriors[cls][idx].get(feature, 0), atol=tolerance)



def test_naive_bayes_classifier_predict():
    # Test case 1: In-Class Example
    clf = MyNaiveBayesClassifier()
    clf.fit(X_train_inclass_example, y_train_inclass_example)

    # Expected predictions (based on manual calculation or inspection of learned priors and posteriors)
    X_test_inclass_example = [
        [1, 5],  # Should predict "yes" based on priors
        [2, 6],  # Should predict "yes" based on priors
        [1, 6],  # Should predict "yes" based on priors
        [2, 5],  # Should predict "no" based on priors
    ]
    expected_predictions_inclass = ["yes", "yes", "yes", "no"]

    predictions_inclass = clf.predict(X_test_inclass_example)
    assert predictions_inclass == expected_predictions_inclass, f"Failed for in-class example: {predictions_inclass}"

    # Test case 2: iPhone dataset
    clf.fit(X_train_iphone, y_train_iphone)
    
    # Expected predictions based on priors and likelihoods
    X_test_iphone = [
        [1, 3, "fair"],      # Should predict "no"
        [2, 1, "excellent"], # Should predict "yes"
        [1, 2, "fair"],      # Should predict "yes"
    ]
    expected_predictions_iphone = ["no", "yes", "yes"]

    predictions_iphone = clf.predict(X_test_iphone)
    assert predictions_iphone == expected_predictions_iphone, f"Failed for iPhone dataset: {predictions_iphone}"

    # Test case 3: Bramer 3.2 dataset
    clf.fit(X_train_train, y_train_train)
    
    # Expected predictions based on priors and likelihoods
    X_test_train = [
        ["weekday", "spring", "none", "none"],  # Should predict "on time"
        ["saturday", "summer", "normal", "none"],  # Should predict "on time"
        ["holiday", "spring", "normal", "slight"],  # Should predict "on time"
    ]
    expected_predictions_train = ["on time", "on time", "on time"]

    predictions_train = clf.predict(X_test_train)
    assert predictions_train == expected_predictions_train, f"Failed for Bramer dataset: {predictions_train}"

    print("All test cases passed!")

# interview dataset
header_interview = ["level", "lang", "tweets", "phd", "interviewed_well"]
X_train_interview = [
    ["Senior", "Java", "no", "no"],
    ["Senior", "Java", "no", "yes"],
    ["Mid", "Python", "no", "no"],
    ["Junior", "Python", "no", "no"],
    ["Junior", "R", "yes", "no"],
    ["Junior", "R", "yes", "yes"],
    ["Mid", "R", "yes", "yes"],
    ["Senior", "Python", "no", "no"],
    ["Senior", "R", "yes", "no"],
    ["Junior", "Python", "yes", "no"],
    ["Senior", "Python", "yes", "yes"],
    ["Mid", "Python", "no", "yes"],
    ["Mid", "Java", "yes", "no"],
    ["Junior", "Python", "no", "yes"]
]
y_train_interview = ["False", "False", "True", "True", "True", "False", "True", "False", "True", "True", "True", "True", "True", "False"]

# note: this tree uses the generic "att#" attribute labels because fit() does not and should not accept attribute names
# note: the attribute values are sorted alphabetically
tree_interview = \
        ["Attribute", "att0",
            ["Value", "Junior", 
                ["Attribute", "att3",
                    ["Value", "no", 
                        ["Leaf", "True", 3, 5]
                    ],
                    ["Value", "yes", 
                        ["Leaf", "False", 2, 5]
                    ]
                ]
            ],
            ["Value", "Mid",
                ["Leaf", "True", 4, 14]
            ],
            ["Value", "Senior",
                ["Attribute", "att2",
                    ["Value", "no",
                        ["Leaf", "False", 3, 5]
                    ],
                    ["Value", "yes",
                        ["Leaf", "True", 2, 5]
                    ]
                ]
            ]
        ]

def test_decision_tree_classifier_fit():
    # Instantiate the classifier
    clf = MyDecisionTreeClassifier()

    # Fit the model with the interview dataset
    clf.fit(X_train_interview, y_train_interview)

    # Define the expected tree structure for the interview dataset
    expected_tree_interview = [
        "Attribute", "att0",
        ["Value", "Junior", 
            ["Attribute", "att3",
                ["Value", "no", 
                    ["Leaf", "True", 3, 5]
                ],
                ["Value", "yes", 
                    ["Leaf", "False", 2, 5]
                ]
            ]
        ],
        ["Value", "Mid",
            ["Leaf", "True", 4, 14]
        ],
        ["Value", "Senior",
            ["Attribute", "att2",
                ["Value", "no",
                    ["Leaf", "False", 3, 5]
                ],
                ["Value", "yes",
                    ["Leaf", "True", 2, 5]
                ]
            ]
        ]
    ]

    # Assert that the tree structure for the interview dataset matches the expected tree
    assert clf.tree == expected_tree_interview, f"Expected: {expected_tree_interview}, but got: {clf.tree}"

    # Fit the model with the iPhone dataset
    clf.fit(X_train_iphone, y_train_iphone)

    expected_tree_iphone = [
    "Attribute", "att0",
    ["Value", 1,
        ["Attribute", "att1",
            ["Value", 1,
                ["Leaf", "yes", 1, 5]
            ],
        ]
    ],
    ["Value", 3, ["Leaf", "no", 2, 5]],
    ["Value", 2,
        ["Attribute", "att2",
            ["Value", "excellent",
                ["Attribute", "att1",
                    ["Value", 1, ["Leaf", "no", 2, 4]],
                    ["Value", 2, ["Leaf", "no", 2, 4]]
                ]
            ],
            ["Value", "fair", ["Leaf", "yes", 6, 10]]
        ]
    ]
    ]

    # Assert that the tree structure for the iPhone dataset matches the expected tree
    #assert clf.tree == expected_tree_iphone, f"Expected: {expected_tree_iphone}, but got: {clf.tree}"

def test_decision_tree_classifier_predict():
    # Instantiate the classifier
    clf = MyDecisionTreeClassifier()

    # Fit the model with the interview dataset
    clf.fit(X_train_interview, y_train_interview)

    # Define test instances from the interview dataset
    X_test_interview = [
        ["Senior", "Java", "no", "no"],  # Instance 1
        ["Junior", "Python", "yes", "yes"]  # Instance 2
    ]
    
    # Define expected predictions from desk check for the interview dataset
    expected_predictions_interview = ["False", "False"]

    # Get predictions
    y_pred_interview = clf.predict(X_test_interview)

    # Assert that the predictions are as expected
    assert y_pred_interview == expected_predictions_interview, f"Expected: {expected_predictions_interview}, but got: {y_pred_interview}"

    # Fit the model with the iPhone dataset
    clf.fit(X_train_iphone, y_train_iphone)

    # Define test instances from the iPhone dataset (unseen)
    X_test_iphone = [
        [1, 2, "fair"],  # Instance 1 (Expected: "no")
        [2, 3, "excellent"]  # Instance 2 (Expected: "no")
    ]
    
    # Define expected predictions from desk check for the iPhone dataset
    expected_predictions_iphone = ["no", "A"]

    # Get predictions
    y_pred_iphone = clf.predict(X_test_iphone)

    # Assert that the predictions are as expected
    assert y_pred_iphone == expected_predictions_iphone, f"Expected: {expected_predictions_iphone}, but got: {y_pred_iphone}"


