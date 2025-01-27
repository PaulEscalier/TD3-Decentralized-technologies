import requests
import numpy as np

# URLs for the models
urls = [
    "https://4f61-89-30-29-68.ngrok-free.app/predict", #Paul
    "https://d5fb-89-30-29-68.ngrok-free.app/predict", #Tristan
    "https://1c43-89-30-29-68.ngrok-free.app/predict", #Maxime
    "https://be19-89-30-29-68.ngrok-free.app/predict" #Cyprien
]

# Input features for the prediction
input_features = [5.1, 3.5, 1.4, 0.2]  # Example input for an Iris flower

# Function to query a model and return predictions
def get_predictions(url, input_features):
    try:
        if "features" in url:  # If the URL expects a "features" parameter
            response = requests.get(url, params={"features": ",".join(map(str, input_features))})
        else:  # If the URL expects individual parameters
            params = {
                "sepal_length": input_features[0],
                "sepal_width": input_features[1],
                "petal_length": input_features[2],
                "petal_width": input_features[3],
            }
            response = requests.get(url, params=params)
        
        response.raise_for_status()  # Raise an exception for HTTP errors
        data = response.json()
        return data.get("probabilities", [])  # Return the probabilities
    except Exception as e:
        print(f"Error querying {url}: {e}")
        return []

# Fetch predictions from all models
predictions = []
for url in urls:
    model_predictions = get_predictions(url, input_features)
    if model_predictions:
        predictions.append(model_predictions)

# Check if we received predictions from all models
if len(predictions) < len(urls):
    print("Some models did not return predictions. Proceeding with available results.")

# Aggregate predictions (average probabilities)
if predictions:
    predictions_array = np.array(predictions)  # Convert to a NumPy array for easier averaging
    averaged_probabilities = np.mean(predictions_array, axis=0)

    # Get the class with the highest probability
    final_class = int(np.argmax(averaged_probabilities))

    # Output the results
    print("Consensus Probabilities:", averaged_probabilities)
    print("Final Predicted Class:", final_class)
else:
    print("No predictions available to aggregate.")