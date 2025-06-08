# Wine Classification Project

## Project Overview

This project demonstrates a complete machine learning lifecycle for **wine classification**, leveraging modern MLOps tools and best practices:

- **Data Management & Versioning:** Using DVC and DAGsHub to track datasets and preprocessing.
- **Model Development & Experiment Tracking:** Using scikit-learn and MLflow for building and tracking models.
- **Model Deployment & API Development:** Building a FastAPI endpoint to serve predictions.
- **CI/CD Automation:** Using GitHub Actions to run tests and deploy the API.
- **Workflow Orchestration (Optional):** Automate preprocessing and retraining with Apache Airflow.

---

## Project Structure
├── api/ # FastAPI app source code
├── data/
│ ├── raw/ # Raw datasets (tracked by DVC)
│ └── processed/ # Preprocessed datasets
├── models/ # Saved trained models (tracked by DVC)
├── preprocess.py # Data preprocessing script
├── train.py # Model training and experiment logging script
├── requirements.txt # Python dependencies
├── dvc.yaml # DVC pipeline stages
├── .github/workflows/ # GitHub Actions workflows for CI/CD
└── README.md # Project documentation


---

## Getting Started

### Prerequisites

- Python 3.10+
- Git
- DVC
- MLflow
- (Optional) Airflow

### Installation

1. Clone the repository:

```bash
git clone <your-repo-url>
cd wine-classification
