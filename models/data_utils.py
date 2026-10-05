from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data():
    df = pd.read_excel(Path(__file__).resolve().parents[1] / "data" / "data.xlsx", sheet_name="Hoja1")
    df = df[df["excluFINAL"] == 1].dropna(subset=["prom"])

    features = [
        "tpopantallatotalhs",
        "WATCHTVbedtime7D01",
        "COMPUbedtime1D01",
        "COMPUbedtime2D01",
        "BMIzscoreCAT",
        "matematica",
        "aplazos"
    ]
    df = df.dropna(subset=features)

    X = df[features].values
    y = df["prom"].values

    X_scaled = StandardScaler().fit_transform(X)
    return train_test_split(X_scaled, y, test_size=0.2, random_state=42)

def get_feature_names():
    return [
        "tpopantallatotalhs",
        "WATCHTVbedtime7D01",
        "COMPUbedtime1D01",
        "COMPUbedtime2D01",
        "BMIzscoreCAT",
        "matematica",
        "aplazos"
    ]
