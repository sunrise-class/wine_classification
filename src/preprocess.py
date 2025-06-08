import pandas as pd
import sys

input_path = 'data/raw/wine.csv'
output_path = 'data/processed/wine_clean.csv'

df = pd.read_csv(input_path)
df = df.dropna()
# More preprocessing...
df.to_csv(output_path, index=False)
