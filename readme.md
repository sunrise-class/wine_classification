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
* ├── api/ # FastAPI app source code
* ├── data/
* │ ├── raw/ # Raw datasets (tracked by DVC)
* │ └── processed/ # Preprocessed datasets
* ├── models/ # Saved trained models (tracked by DVC)
* ├── preprocess.py # Data preprocessing script
* ├── train.py # Model training and experiment logging script
* ├── requirements.txt # Python dependencies
* ├── dvc.yaml # DVC pipeline stages
* ├── .github/workflows/ # GitHub Actions workflows for CI/CD
* └── README.md # Project documentation


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
git clone https://github.com/sunrise-class/wine_classification.git
cd wine-classification
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```


3. Initialize DVC and add remote storage (if not already done):

```bash
dvc init
dvc remote add -d myremote <your-remote-storage-url>
```

4. Pull datasets and models tracked by DVC
```bash
dvc pull
```


## Usage

### Data Preprocessing

Run the preprocessing script to clean and transform the raw data:

```bash
python src/preprocess.py
```

Preprocessed data is saved to data/processed and tracked by DVC.

Model Training and Experiment Tracking
Train your classification model and log experiments with MLflow:

```bash
python src/train.py
python src/model_selection.py
python src/save_wine_data.py
```

Launch MLflow UI locally to visualize experiments:

```bash
mlflow ui
```
Visit http://127.0.0.1:5000 in your browser.

Running the API Server
Serve the trained model via FastAPI:

```bash
uvicorn api.main:app --reload
```

API documentation is available at:
http://127.0.0.1:8000/docs

## CI/CD Automation with GitHub Actions

github ci cd path: .github/workflows/ci-cd.yml

Testing: Automatically runs unit tests and integration tests on every push to the main branch.

Data & Model Sync: Pulls latest datasets and models with DVC using DAGsHub token.

Deployment: Deploys the FastAPI app on merge to the main branch (deployment step placeholder — customize for your target platform).