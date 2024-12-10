import requests # a lib for making http requests
import json # a lib for working with json

url = "http://localhost:5001/predict?TEAM=Duke&ADJOE=125.2&ADJDE=90.6&EFF_O=0.9764&EFF_D=56.6&YEAR=2015"

response = requests.get(url=url)

# first thing, check the response's status_code
# https://developer.mozilla.org/en-US/docs/Web/HTTP/Methods
# https://developer.mozilla.org/en-US/docs/Web/HTTP/Status#successful_responses
print(response.status_code)
if response.status_code == 200:
    # STATUS OK
    # we can extract the prediction from the response's JSON text
    json_object = json.loads(response.text)
    print(json_object)
    pred = json_object["prediction"]
    print("prediction:", pred)