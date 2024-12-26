import requests

# URL of the Flask API
url = "http://127.0.0.1:5000/search"

# Test payload
payload = {
    "query": "donut"
}

# Send a POST request to the API
response = requests.post(url, json=payload)

# Print the response
if response.status_code == 200:
    print("Test Successful!")
    print("Response JSON:", response.json())
else:
    print(f"Test Failed! Status Code: {response.status_code}")
    print("Error Response:", response.text)
