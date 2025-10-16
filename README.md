# MLOps Week 4 Assignment

## Overview  
This project demonstrates an **end-to-end MLOps workflow** using the **Iris dataset**.  
It integrates:
- **DVC** for data and model versioning  
- **GitHub Actions** for Continuous Integration (CI)  
- **Pytest** for data and model validation  
- **CML** for automated reporting in GitHub comments  

---

## Objectives  
- Setup a GitHub repository with `dev` and `main` branches  
- Create unit tests for data validation and model evaluation  
- Configure **CI** to pull data and model from **DVC (GCS remote)**  
- Automate testing and reporting using **CML**  

---

## Files and Folders  
## MLOps Project Structure

This structure highlights a typical organization for a machine learning project, leveraging **DVC** (Data Version Control) for tracking data and models, and a **CI/CD pipeline** for automation.

### Files and Folders Tree

```text
MLOps-week4/
├── data/                                 # Iris dataset (tracked by DVC)
├── artifacts/                            # Trained model (tracked by DVC)
├── tests/
│   ├── test_data_validation.py           # Validates the dataset
│   └── test_evaluation.py                # Tests model accuracy
├── main.py                               # Model training script
├── augment_data.py                       # Data augmentation script
├── requirements.txt                      # Project dependencies
└── .github/
    └── workflows/
        └── ci-dev.yml                    # CI pipeline configuration

```

## Key Commands  
```bash
dvc pull -r gcsremote
pytest -q
git commit --allow-empty -m "Trigger CI"
git push origin dev
```


