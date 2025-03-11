import pandas as pd
import os

url = "https://datascientest-mlops.s3.eu-west-1.amazonaws.com/mlops_dvc_fr/raw.csv"
df = pd.read_csv(url)
os.makedirs("data/raw", exist_ok=True)
df.to_csv("data/raw/raw.csv", index=False)