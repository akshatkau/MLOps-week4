#!/bin/bash

pip install -r requirements.txt --quiet
pip install google-cloud-aiplatform --upgrade --quiet

BUCKET_NAME="21f3000376-week2"
REGION="us-central1"
gsutil mb -l "$REGION" -p "gen-lang-client-0761066410" "gs://$BUCKET_NAME" || echo "Bucket already exists."

echo "Using bucket: gs://$BUCKET_NAME"

mkdir -p artifacts

python main.py

echo "Uploading model.joblib to GCS..."
gsutil cp artifacts/model.joblib "gs://$BUCKET_NAME/models/iris-classifier-week-2/"

echo "Upload complete!"
