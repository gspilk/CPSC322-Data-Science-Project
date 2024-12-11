#to do: create the flask which allows users to predict using the random forests algorithm.
import pickle
from flask import Flask, request, jsonify
import numpy as np

app = Flask(__name__)

def load_model():
    # unpickle header and tree in tree.p
    infile = open("flaskapp/tree.p", "rb")
    header, tree = pickle.load(infile)
    infile.close()
    return header, tree


@app.route("/")
def index():
    return """

        <h1>Welcome to the March Madness Predictor App</h1>
        <img src="https://i.bleacherreport.net/images/team_logos/328x328/gonzaga_basketball.png?canvas=492,328" alt="Gonzaga Bulldogs">
        
        <!-- Example button to trigger a prediction or form submission -->
        <form action="https://march-madness-tournament-predictor.onrender.com/" method="get">
            <label for="TEAM">Team:</label>
            <input type="text" id="TEAM" name="TEAM"><br><br>
            
            <label for="ADJOE">ADJOE:</label>
            <input type="text" id="ADJOE" name="ADJOE"><br><br>

            <label for="ADJDE">ADJDE:</label>
            <input type="text" id="ADJDE" name="ADJDE"><br><br>

            <label for="EFF_O">EFF_O:</label>
            <input type="text" id="EFF_O" name="EFF_O"><br><br>

            <label for="EFF_D">EFF_D:</label>
            <input type="text" id="EFF_D" name="EFF_D"><br><br>

            <label for="YEAR">YEAR:</label>
            <input type="text" id="YEAR" name="YEAR"><br><br>

            <!-- Button to submit the form -->
            <button type="submit">Make Prediction</button>
        </form>
    """, 200


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
