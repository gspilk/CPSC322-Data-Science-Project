import requests

url = "https://march-madness-tournament-predictor.onrender.com"

# Define query parameters for prediction
params = {
    "TEAM": "Duke",    # Team name
    "ADJOE": 125.2,    # Offensive efficiency
    "ADJDE": 90.6,     # Defensive efficiency
    "YEAR": 2015       # Year of tournament
}

# Send the GET request
response = requests.get(url, params=params)

# Handle the response
print(f"Status Code: {response.status_code}")
if response.status_code == 200:
    json_object = response.json()
    print(f"Prediction for {json_object['team']} ({json_object['year']}): {json_object['prediction']}")
else:
    print("Error:", response.text)
