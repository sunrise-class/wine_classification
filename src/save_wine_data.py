# src/save_wine_data.py
import pandas as pd
from sklearn.datasets import load_wine

def save_wine_data(path='data/raw/wine.csv'):
    data = load_wine(as_frame=True)
    df = pd.concat([data.data, pd.Series(data.target, name="target")], axis=1)
    df.to_csv(path, index=False)
    print(f"Wine dataset saved to {path}")

if __name__ == "__main__":
    save_wine_data()