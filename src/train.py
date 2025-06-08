import mlflow
import mlflow.sklearn
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from model_selection import select_and_save_best_model
import joblib
# Load wine dataset
data = load_wine()
X = data.data
y = data.target

# Scale features for better performance
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the dataset into training and testing sets (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

mlflow.set_tracking_uri("https://dagshub.com/pechnary15/wine-classification.mlflow")
mlflow.set_tracking_uri("https://pechnary15:384668c8ba37f965f036d7f22beb93e75cb56287@dagshub.com/pechnary15/wine-classification.mlflow")


# Define function to log models, hyperparameters, and metrics
def log_model(model_name, model, params, metrics):
    with mlflow.start_run(run_name=model_name):  # Set the run name to model name
        # Log model parameters (hyperparameters)
        for param, value in params.items():
            mlflow.log_param(param, value)

        # Log metrics
        for metric, value in metrics.items():
            mlflow.log_metric(metric, value)

        # Log model as artifact
        mlflow.sklearn.log_model(model, model_name)

        # Print the model name to show in the terminal
        print(f"Model '{model_name}' is logged with the following metrics:")
        for metric, value in metrics.items():
            print(f"{metric}: {value}")
        print("----")

# Logistic Regression Model (Classification)
log_params_lr = {'C': 1.0, 'max_iter': 1000}
log_model_lr = LogisticRegression(C=1.0, max_iter=1000, random_state=42)

# Train Logistic Regression Model
log_model_lr.fit(X_train, y_train)
y_pred_lr = log_model_lr.predict(X_test)

# Evaluate the model
accuracy_lr = accuracy_score(y_test, y_pred_lr)
precision_lr = precision_score(y_test, y_pred_lr, average='weighted')
recall_lr = recall_score(y_test, y_pred_lr, average='weighted')
f1_lr = f1_score(y_test, y_pred_lr, average='weighted')

# Log Logistic Regression Model in MLflow
log_model('Logistic Regression', log_model_lr, log_params_lr, {
    'accuracy': accuracy_lr,
    'precision': precision_lr,
    'recall': recall_lr,
    'f1_score': f1_lr
})

# SVM Model (Classification)
log_params_svm = {'C': 1.0, 'kernel': 'linear'}
log_model_svm = SVC(C=1.0, kernel='linear', random_state=42)

# Train SVM Model
log_model_svm.fit(X_train, y_train)
y_pred_svm = log_model_svm.predict(X_test)

# Evaluate the model
accuracy_svm = accuracy_score(y_test, y_pred_svm)
precision_svm = precision_score(y_test, y_pred_svm, average='weighted')
recall_svm = recall_score(y_test, y_pred_svm, average='weighted')
f1_svm = f1_score(y_test, y_pred_svm, average='weighted')

# Log SVM Model in MLflow
log_model('SVM Classifier', log_model_svm, log_params_svm, {
    'accuracy': accuracy_svm,
    'precision': precision_svm,
    'recall': recall_svm,
    'f1_score': f1_svm
})

# Random Forest Regressor Model
log_params_rf = {'n_estimators': 100, 'max_depth': 10}
log_model_rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)

# Train Random Forest Regressor Model
log_model_rf.fit(X_train, y_train)
y_pred_rf = log_model_rf.predict(X_test)

# Evaluate the model
mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

# Log Random Forest Regressor Model in MLflow
log_model('Random Forest Regressor', log_model_rf, log_params_rf, {
    'mse': mse_rf,
    'r2_score': r2_rf
})

print("\nAll models have been logged to MLflow!")
# Evaluate SVM
f1_svm = f1_score(y_test, y_pred_svm, average='weighted')

# Select and save the best model based on F1 scores
select_and_save_best_model(f1_lr, f1_svm, log_model_lr, log_model_svm)
joblib.dump(scaler, 'models/scaler_all_features.joblib')