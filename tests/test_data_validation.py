import pandas as pd

def test_no_missing_values():
    """Ensure the dataset has no missing values."""
    df = pd.read_csv("data/iris.csv")
    assert df.isnull().sum().sum() == 0, "Dataset contains missing values!"

def test_valid_species_labels():
    """Ensure species column has only expected labels."""
    df = pd.read_csv("data/iris.csv")
    valid_labels = {"setosa", "versicolor", "virginica"}
    assert set(df["species"].unique()).issubset(valid_labels), "Unexpected species label found!"
