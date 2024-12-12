import pickle
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load the pickled tree
def load_tree():
    with open("flaskapp/tree.p", "rb") as infile:
        header, tree = pickle.load(infile)
    return header, tree

# Prediction logic using a decision tree
def predict_tree(tree, instance, header):
    if tree[0] == "Leaf":
        return tree[1]  # Return the prediction from the leaf
    elif tree[0] == "Attribute":
        attribute = tree[1]
        attr_index = header.index(attribute)
        instance_value = instance[attr_index]

        for branch in tree[2:]:
            if branch[0] == "Value" and branch[1] == instance_value:
                return predict_tree(branch[2], instance, header)
    return None  # If no matching branch is found

@app.route("/")
def index():
    return """
    <h1>Welcome to the March Madness Predictor App</h1>
    <img src="https://i.bleacherreport.net/images/team_logos/328x328/gonzaga_basketball.png?canvas=492,328" alt="Gonzaga Bulldogs">

    <form action="/predict" method="get">
        <label for="TEAM">TEAM:</label>
        <input type="text" id="TEAM" name="TEAM"><br><br>
        
        <label for="ADJOE">ADJOE:</label>
        <input type="text" id="ADJOE" name="ADJOE"><br><br>

        <label for="ADJDE">ADJDE:</label>
        <input type="text" id="ADJDE" name="ADJDE"><br><br>

        <label for="YEAR">YEAR:</label>
        <input type="text" id="YEAR" name="YEAR"><br><br>

        <button type="submit">Make Prediction</button>
    </form>
    """, 200

@app.route("/predict")
def predict():
    # Parse query parameters
    try:
        team = request.args.get("TEAM")
        adjoe = float(request.args.get("ADJOE"))
        adjde = float(request.args.get("ADJDE"))
        year = int(request.args.get("YEAR"))
    except (TypeError, ValueError):
        return "Invalid input parameters. Please provide valid values.", 400

    # Create instance for prediction
    instance = [team, adjoe, adjde, year, None]  # `None` is placeholder for POSTSEASON

    # Load the tree and predict
    header, tree = load_tree()
    prediction = predict_tree(tree, instance, header)

    return jsonify({
        "prediction": prediction
    }), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=False)
