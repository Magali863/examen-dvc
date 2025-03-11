# src/data/split_data.py
import pandas as pd
import os
from sklearn.model_selection import train_test_split

# Charger les données
data = pd.read_csv("data/raw/raw.csv")

# Séparer les features (X) et la cible (y)
X = data.drop("silica_concentrate", axis=1)  
y = data["silica_concentrate"]  

# Split en train et test (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Créer le dossier processed si nécessaire
os.makedirs("data/processed", exist_ok=True)

# Sauvegarder les fichiers
X_train.to_csv("data/processed/X_train.csv", index=False)
X_test.to_csv("data/processed/X_test.csv", index=False)
y_train.to_csv("data/processed/y_train.csv", index=False)
y_test.to_csv("data/processed/y_test.csv", index=False)

print("Données splittées et sauvegardées dans data/processed.")

