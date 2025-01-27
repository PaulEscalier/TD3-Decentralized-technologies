# Example with scikit-learn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from flask import Flask, request, jsonify
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Charger le dataset iris
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)  # Caractéristiques
y = pd.Series(iris.target)

# Fractionner les données
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Mise à l'échelle (scaling)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Entraîner un modèle de régression logistique
model_lr = LogisticRegression()
model_lr.fit(X_train, y_train)
y_pred_lr = model_lr.predict(X_test)
accuracy_lr = accuracy_score(y_test, y_pred_lr)
print("Logistic Regression Accuracy:", accuracy_lr)

# Initialisation de l'application Flask
app = Flask(__name__)

# Modèle sélectionné
selected_model = model_lr  # On utilise Logistic Regression

@app.route('/')
def hello_world():
    return 'Hello, World!'

@app.route('/predict', methods=['GET'])
def predict():
    try:

        sepal_length = float(request.args.get('sepal_length'))
        sepal_width = float(request.args.get('sepal_width'))
        petal_length = float(request.args.get('petal_length'))
        petal_width = float(request.args.get('petal_width'))

        # Construire le tableau des caractéristiques
        features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

        # Effectuer la prédiction (probabilités pour chaque classe)
        probabilities = selected_model.predict_proba(features)


        # Préparer une réponse JSON plus sobre
        response = {
            'probabilities': [round(float(prob), 5) for prob in probabilities[0]]  # Limiter à 10 décimales
        }
        return jsonify(response)
    except Exception as e:
        return jsonify({'error': str(e)})

# Lancer l'application Flask
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
