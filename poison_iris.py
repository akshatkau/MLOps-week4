import pandas as pd
import numpy as np
import sys
import os

# Usage: python poison_iris.py 5   (for 5% noise)
percent = int(sys.argv[1])

data_path = "data/iris.csv"
df = pd.read_csv(data_path)

# Number of rows to poison
n = int(len(df) * percent / 100)

# Select rows to poison
rows = np.random.choice(df.index, n, replace=False)

# Copy original
poisoned = df.copy()

# Add random numeric noise to selected rows
for col in poisoned.columns:
    if poisoned[col].dtype != object:
        noise = np.random.normal(0, 1, n)
        poisoned.loc[rows, col] = poisoned.loc[rows, col] + noise

# Save poisoned dataset
out_path = f"data/iris_poison_{percent}.csv"
poisoned.to_csv(out_path, index=False)

print(f"Poisoned {n} rows ({percent}%) and saved to {out_path}")
