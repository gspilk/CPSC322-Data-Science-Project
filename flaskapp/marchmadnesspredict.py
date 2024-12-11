import requests # a lib for making http requests
import json # a lib for working with json

url = "https://march-madness-champion-predictor.onrender.com"

# Make a GET request to the /predict endpoint with query parameters
response = requests.get(url, params={
    "TEAM": "Duke", 
    "ADJOE": "125.2",
    "ADJDE": "90.6",
    "EFF_O": "0.9764",
    "EFF_D": "56.6",
    "YEAR": "2015"
})

# Check if the request was successful (status code 200)
print(response.status_code)
if response.status_code == 200:
    # STATUS OK
    # Extract the prediction from the response's JSON text
    json_object = json.loads(response.text)
    print(json_object)
    pred = json_object["prediction"]
    print("Prediction:", pred)
else:
    print("Error:", response.status_code)
