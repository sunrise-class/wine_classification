from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Load the trained model and scaler
model = joblib.load('/models/Logistic Regression_best_model.joblib')  # Path to the model
scaler = joblib.load('/models/scaler_all_features.joblib')  # Path to the scaler

# Initialize FastAPI app
app = FastAPI()


# Define the request body structure for FastAPI to accept all 13 features
class WineFeatures(BaseModel):
    alcohol: float
    malic_acid: float
    ash: float
    alcalinity_of_ash: float
    magnesium: float
    total_phenols: float
    flavonoids: float
    nonflavanoid_phenols: float
    proanthocyanins: float
    color_intensity: float
    hue: float
    od280_od315_of_diluted_wine: float
    proline: float

# Map numeric predictions to text labels
class_labels = {
    0: "Class 0: Wine Type 1",
    1: "Class 1: Wine Type 2"
}

# Define a predict route that uses the model to predict
@app.post("/predict")
def predict(features: WineFeatures):
    # Extract the features from the request and convert to a numpy array
    feature_values = np.array([
        features.alcohol,
        features.malic_acid,
        features.ash,
        features.alcalinity_of_ash,
        features.magnesium,
        features.total_phenols,
        features.flavonoids,
        features.nonflavanoid_phenols,
        features.proanthocyanins,
        features.color_intensity,
        features.hue,
        features.od280_od315_of_diluted_wine,
        features.proline
    ]).reshape(1, -1)  # Reshape to match the expected input format

    # Scale the input features using the same scaler that was used during training
    feature_values_scaled = scaler.transform(feature_values)

    # Make the prediction using the trained model
    prediction = model.predict(feature_values_scaled)

    # Map the prediction to the class label text
    prediction_text = class_labels[int(prediction[0])]

    # Return the prediction text
    return {"prediction": prediction_text}
