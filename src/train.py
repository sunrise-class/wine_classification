import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import joblib
from src.data_loader import load_data
from src.evaluate import get_metrics

X_train, X_test, y_train, y_test = load_data()
models = {
    "Logistic Regression": LogisticRegression(max_iter=200),
    "Random Forest": RandomForestClassifier(n_estimators=100),
    "SVM": SVC(probability=True)
}

best_score = 0
best_model = None

for name, model in models.items():
    with mlflow.start_run(run_name=name):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        metrics = get_metrics(y_test, y_pred)
        mlflow.log_params(model.get_params())
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, "model")

        if metrics["f1_score"] > best_score:
            best_score = metrics["f1_score"]
            best_model = model

# Save best model
joblib.dump(best_model, "models/best_model.pkl")