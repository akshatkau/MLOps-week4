## Project Structure

- **`data/iris.csv`**  
  Dataset used for training. Contains 4 numerical features (`sepal_length`, `sepal_width`, `petal_length`, `petal_width`) and one target label (`species`).

- **`main.py`**  
  Training script that:
  - Loads the Iris dataset.  
  - Splits data into train/test sets.  
  - Trains a `DecisionTreeClassifier`.  
  - Evaluates accuracy.  
  - Saves the trained model into the `artifacts/` directory as `model.joblib`.  
  - Stores evaluation metrics in `metrics.csv`.

- **`alter_data.py`**  
  A helper script that modifies the dataset by randomly selecting 50 rows.  
  This simulates updated/augmented data for retraining.

- **`main.sh`**  
  Shell script to automate workflow:
  - Installs required dependencies.  
  - Creates (or reuses) a Google Cloud Storage bucket.  
  - Executes `main.py` to train the model.  
  - Uploads the saved model artifact (`artifacts/model.joblib`) to the GCS bucket.

- **`requirements.txt`**  
  Contains Python dependencies required to run the pipeline:
# rerun after dvc-gs push
