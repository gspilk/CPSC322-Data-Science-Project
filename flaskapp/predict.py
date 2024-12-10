#to do: create the flask which allows users to predict using the random forests algorithm.
import pickle
from flask import Flask, request, jsonify

app = Flask(__name__)

def load_model():
    # unpickle header and tree in tree.p
    infile = open("flaskapp/tree.p", "rb")
    header, tree = pickle.load(infile)
    infile.close()
    return header, tree

def predict(self, X_test):
    """Makes predictions for test instances in X_test.

    Args:
        X_test (list of list of obj): The list of testing samples
            The shape of X_test is (n_test_samples, n_features)

    Returns:
        y_predicted (list of obj): The predicted target y values (parallel to X_test)
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

@app.route("/")
def index():
    return "<h1>Welcome to the March Madness Predictor App<h1>",200

# # lets add a route for the /predict endpoint
@app.route("/predict")
def predict():
    # lets parse the unseen instance values from the query string
    # they are in the request object
    team = request.args.get("TEAM") # defaults to None
    adjoe = request.args.get("ADJOE")
    adjde = request.args.get("ADJDE")
    eff_o = request.args.get("EFF_O")
    eff_d = request.args.get("EFF_D")
    year = request.args.get("YEAR")

    instance = [team,adjoe,adjde,eff_o,eff_d,year]
    header, tree = load_model()
    # lets make a prediction!
    pred = predict(header, tree, instance)
    if pred is not None:
        return jsonify({"prediction": pred}), 200
    # something went wrong!!
    return "Error making a prediction", 400


if __name__ == "__main__":
   app.run(host="0.0.0.0",port=5001, debug=False) 