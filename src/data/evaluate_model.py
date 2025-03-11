# src/data/evaluate_model.py
import pandas as pd
import pickle
import json
from sklearn.metrics import mean_squared_error, r2_score

# Charger les données
X_test_scaled = pd.read_csv("data/processed/X_test_scaled.csv")
y_test = pd.read_csv("data/processed/y_test.csv")

# Charger le modèle entraîné
with open("models/models/trained_model.pkl", "rb") as f:
    model = pickle.load(f)

# Faire des prédictions
y_pred = model.predict(X_test_scaled)

# Calculer les métriques
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Sauvegarder les métriques
metrics = {"mse": mse, "r2": r2}
with open("metrics/scores.json", "w") as f:
    json.dump(metrics, f, indent=4)

# Sauvegarder les prédictions
predictions = pd.DataFrame({"y_true": y_test.values.ravel(), "y_pred": y_pred})
predictions.to_csv("data/predictions.csv", index=False)

print("Évaluation terminée. Métriques sauvegardées dans metrics/scores.json et prédictions dans data/predictions.csv.")