# model_selection.py

import os
import joblib
import mlflow
import mlflow.sklearn

def select_and_save_best_model(f1_lr, f1_svm, log_model_lr, log_model_svm):
    best_model = None
    best_score = -float('inf')

    # Compare F1 scores for Logistic Regression and SVM
    if f1_lr > best_score:
        best_score = f1_lr
        best_model = log_model_lr
        best_model_name = 'Logistic Regression'

    if f1_svm > best_score:
        best_score = f1_svm
        best_model = log_model_svm
        best_model_name = 'SVM Classifier'

    # Create the directory for saving the model if it doesn't exist
    model_save_path = 'models/'
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)

    # Save the best model in a different file
    best_model_filename = os.path.join(model_save_path, f'{best_model_name}_best_model.joblib')
    joblib.dump(best_model, best_model_filename)
    print(f"Best model '{best_model_name}' saved as {best_model_filename}")

