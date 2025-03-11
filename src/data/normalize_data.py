# src/data/normalize_data.py
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
import pickle

# Charger les données
X_train = pd.read_csv("data/processed/X_train.csv")
X_test = pd.read_csv("data/processed/X_test.csv")

# Normalisation avec StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convertir en DataFrame pour sauvegarder
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)

# Sauvegarder les données normalisées
os.makedirs("data/processed", exist_ok=True)
X_train_scaled.to_csv("data/processed/X_train_scaled.csv", index=False)
X_test_scaled.to_csv("data/processed/X_test_scaled.csv", index=False)

# Sauvegarder le scaler pour une utilisation future
with open("models/models/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("Données normalisées et sauvegardées dans data/processed.")
