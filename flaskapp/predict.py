#to do: create the flask which allows users to predict using the random forests algorithm.
import pickle
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route("/")
def index():
    return "<h1>Welcome to the March Madness Predictor App<h1>",200

# @app.route("/predict")
# def predict():


if __name__ == "__main__":
   app.run(host="0.0.0.0",port=5001, debug=True) 