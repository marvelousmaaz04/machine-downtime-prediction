# Machine Downtime Prediction API

This Flask API predicts machine downtime in a manufacturing environment.

## Setup

1. **Clone:** `git clone https://github.com/marvelousmaaz04/machine-downtime-prediction.git`
2. **Install:** `pip install -r requirements.txt`

## File Structure

1. **Machine_Downtime_Prediction.ipynb**: This notebook contains all the data analysis and predictive modeling. Multiple models including `LogisticRegression`, `DecisionTreeClassifier`, `RandomForestClassifier` and `XGBClassifier` were tested. After hyperparameter tuning `DecisionTreeClassifier` emerged as the best model.

2. **best_models/**: This directory contains the best models saved from the notebook.

3. **restful_api.py**: This file is the Flask API for interacting with the `DecisionTreeClassifier` model.

## Run

1. **Navigate:** `cd machine-downtime-prediction>`
2. **Run:** `python restful_api.py`. The development server will start at `https://localhost:5000`

**API Endpoints:**

* **`/upload` (POST):** Use this endpoint to upload custom CSV data with key columns such as `Machine_ID`, `Temperature`, `Run_Time`, `Downtime_Flag`.
* **`/train` (POST):** Train the model on the uploaded CSV data.
* **`/predict` (POST):** Make predictions on test data. As the API also contains the `pre-trained` model, this endpoint can function directly.

## Using Postman to test the API

**Example `/upload` Request:**

![image](https://github.com/user-attachments/assets/67a7db0e-d702-4535-bc9b-24d6510b3e5c)


```form-data
1. Select type 'File' and put the key as `file`
2. Select the required dataset containing key columns ``Machine_ID``, ``Temperature``, ``Run_Time``, ``Downtime_Flag``
```

**Example `/upload` Response:**
```json
{
    "columns": [
        "Machine_ID",
        "Temperature",
        "Run_Time",
        "Downtime_Flag"
    ],
    "message": "File uploaded successfully"
}
```

**Example `/train` Request:**

![image](https://github.com/user-attachments/assets/aee31cac-ff7d-48e9-8fe1-99695453ea3f)

**Example `/train` Response:**

```json
{
    "accuracy": 0.82,
    "f1-score-weighted": 0.77,
    "message": "DecisionTreeClassifier trained and evaluated successfully",
    "metrics": {
        "0": {
            "f1-score": 0.8898170246618934,
            "precision": 0.8032315978456014,
            "recall": 0.9973250111457869,
            "support": 2243.0
        },
        "1": {
            "f1-score": 0.43004115226337447,
            "precision": 0.9720930232558139,
            "recall": 0.2760898282694848,
            "support": 757.0
        },
        "accuracy": 0.8153333333333334,
        "macro avg": {
            "f1-score": 0.6599290884626339,
            "precision": 0.8876623105507077,
            "recall": 0.6367074197076359,
            "support": 3000.0
        },
        "weighted avg": {
            "f1-score": 0.7738002461933338,
            "precision": 0.8458409641907784,
            "recall": 0.8153333333333334,
            "support": 3000.0
        }
    }
}
```

**Example `/predict` Request:**

![image](https://github.com/user-attachments/assets/94059bc7-d6c4-43bf-8564-8b6bc727e8ce)

```json
{
    "Temperature": 100, 
    "Run_Time": 400
}
```

**Example `/predict` Response:**
```json
{
    "Confidence": 1.0,
    "Downtime": "Yes"
}
```
