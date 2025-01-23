from flask import Flask, request, jsonify
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np

app = Flask(__name__)

# Load the saved model
model_data = joblib.load("best_models/tuned_decision_tree_with_params.joblib")
model = model_data["model"]  # Extract the model

# Placeholder for uploaded dataset
uploaded_data = None


# Upload Endpoint
@app.route("/upload", methods=["POST"])
def upload_data():
    global uploaded_data
    file = request.files["file"]
    if not file:
        return jsonify({"error": "No file provided"}), 400
    
    # Read the CSV file into a Pandas DataFrame
    uploaded_data = pd.read_csv(file)
    return jsonify({"message": "File uploaded successfully", "columns": list(uploaded_data.columns)}), 200


@app.route("/train", methods=["POST"])
def train_model():
    global uploaded_data
    if uploaded_data is None:
        return jsonify({"error": "No data uploaded"}), 400

    # Extract features and target
    X = uploaded_data[["Temperature", "Run_Time"]]
    y = uploaded_data["Downtime_Flag"]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Train the model on the training set
    model.fit(X_train, y_train)

    # Evaluate the model on the testing set
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)

    # Get model parameters
    model_params = model.get_params()

    # Return response with metrics and parameters
    return jsonify({
        "accuracy": np.round(report['accuracy'],2),
        'f1-score-weighted': np.round(report['weighted avg']['f1-score'],2),
        "metrics": report,
        "message": "DecisionTreeClassifier trained and evaluated successfully",
    }), 200

# Predict Endpoint
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No input data provided"}), 400

    # Extract features from JSON input
    temperature = data.get("Temperature")
    runtime = data.get("Run_Time")
    if temperature is None or runtime is None:
        return jsonify({"error": "Missing required fields: 'Temperature', 'Run_Time'"}), 400

    # Make prediction
    predict_dict = {
        'Temperature': [temperature],
        'Run_Time': [runtime]
    }
    predict_df = pd.DataFrame(predict_dict)
    print(predict_df)
    # prediction = model.predict([[temperature, runtime]])[0]
    prediction = model.predict(predict_df)
    confidence = model.predict_proba(predict_df)[0][1]  # Probability of Downtime = Yes

    # Format response
    result = {
        "Confidence": round(confidence, 2),
        "Downtime": "Yes" if prediction == 1 else "No"
    }
    return jsonify(result), 200


if __name__ == "__main__":
    app.run(debug=True)
