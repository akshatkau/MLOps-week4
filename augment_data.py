import pandas as pd

iris_df = pd.read_csv("data/iris.csv")

augmented_df = iris_df.sample(n=50, random_state=101)

augmented_df.to_csv("data/iris.csv", index=False)

print("Augmented dataset created with 50 random samples and saved to data/iris.csv")
