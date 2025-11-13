# Week 8 – Data Poisoning, MLflow Tracking, and CI Validation

This branch implements controlled data poisoning on the IRIS dataset, retrains the model under different corruption levels, and evaluates the impact using MLflow. The pipeline is integrated with DVC for data versioning and GitHub Actions for CI validation.

## Overview

Week 8 covers:
- Generating poisoned datasets at 5%, 10%, and 50% noise levels
- Training a Decision Tree classifier on each dataset
- Tracking parameters, metrics, and artifacts using MLflow
- Validating reproducibility through DVC and GitHub Actions CI

## Data Poisoning

The script `poison_iris.py` injects Gaussian noise into a percentage of rows:

```bash
python poison_iris.py 5
python poison_iris.py 10
python poison_iris.py 50
```
Outputs:

iris_poison_5.csv

iris_poison_10.csv

iris_poison_50.csv

## Training and MLflow Logging

Training supports dataset selection:
```bash
python main.py --data data/iris.csv
python main.py --data data/iris_poison_5.csv
python main.py --data data/iris_poison_10.csv
python main.py --data data/iris_poison_50.csv
```

Each run logs:

Accuracy

Parameters

Model artifact (model.joblib)

Metrics file (metrics.csv)

## MLflow UI can be launched with:
```bash
mlflow ui --port 5001
```
