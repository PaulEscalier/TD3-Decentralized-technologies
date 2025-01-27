import requests
import numpy as np
import json

# URLs for the models
urls = [
    "https://28d0-89-30-29-68.ngrok-free.app/predict",  # Paul
    "https://d5fb-89-30-29-68.ngrok-free.app/predict",  # Tristan
    "https://1c43-89-30-29-68.ngrok-free.app/predict",  # Maxime
    "https://db3a-89-30-29-68.ngrok-free.app/predict",  # Cyprien
]

# Initialize balances and weights (stored in JSON)
database_file = "model_balances.json"
initial_deposit = 1000.0  # Initial deposit in euros

# Input features for the prediction
input_features = [5.1, 3.5, 1.4, 0.2]  # Example input for an Iris flower

# Initialize database if it doesn't exist
def initialize_database():
    try:
        with open(database_file, "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        data = {url: {"balance": initial_deposit, "weight": 1.0} for url in urls}
        with open(database_file, "w") as f:
            json.dump(data, f, indent=4)
    return data

# Save database to file
def save_database(data):
    with open(database_file, "w") as f:
        json.dump(data, f, indent=4)

# Function to query a model and return predictions
def get_predictions(url, input_features):
    try:
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

# Adjust weights and apply slashing based on model performance
def adjust_weights_and_slash(database, predictions, consensus_class, slashing_amount=50.0, reward_amount=10.0):
    for i, url in enumerate(urls):
        predicted_class = np.argmax(predictions[i])
        if predicted_class == consensus_class:
            # Reward accurate predictions
            database[url]["balance"] += reward_amount
            database[url]["weight"] += 0.1
        else:
            # Slash inaccurate predictions
            database[url]["balance"] -= slashing_amount
            database[url]["weight"] -= 0.1

        # Ensure weights stay non-negative
        database[url]["weight"] = max(0.1, database[url]["weight"])  # Minimum weight
        database[url]["balance"] = max(0.0, database[url]["balance"])  # Minimum balance

# Fetch predictions from all models
def fetch_all_predictions(input_features):
    predictions = []
    for url in urls:
        model_predictions = get_predictions(url, input_features)
        if model_predictions:
            predictions.append(model_predictions)
    return predictions

# Main logic
if _name_ == "_main_":
    # Load or initialize the database
    model_database = initialize_database()

    # Fetch predictions from all models
    predictions = fetch_all_predictions(input_features)

    # Check if we received predictions from all models
    if len(predictions) < len(urls):
        print("Some models did not return predictions. Proceeding with available results.")

    # Aggregate predictions using weights
    if predictions:
        # Fetch weights from the database
        weights = np.array([model_database[url]["weight"] for url in urls])

        # Calculate weighted probabilities
        predictions_array = np.array(predictions)
        weighted_predictions = np.average(predictions_array, axis=0, weights=weights)

        # Get the consensus class (highest probability)
        consensus_class = int(np.argmax(weighted_predictions))

        # Adjust weights and slash balances
        adjust_weights_and_slash(model_database, predictions, consensus_class)

        # Output results
        print("Weighted Consensus Probabilities:", weighted_predictions)
        print("Consensus Final Predicted Class:", consensus_class)

        # Show balances and weights
        for url in urls:
            print(f"Model {url} - Balance: €{model_database[url]['balance']:.2f}, Weight: {model_database[url]['weight']:.2f}")

        # Save updated database
        save_database(model_database)
    else:
        print("No predictions available to aggregate.")