from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd

app = Flask(__name__)

# Load the model and preprocessing objects
with open("model/rf_model.pkl", "rb") as f:
    rf_model = pickle.load(f)

with open("model/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("model/columns.pkl", "rb") as f:
    train_columns = pickle.load(f)

with open("model/num_features.pkl", "rb") as f:
    num_features = pickle.load(f)

with open("model/cat_features.pkl", "rb") as f:
    cat_features = pickle.load(f)


def preprocess_input(user_input):
    """
    Preprocess user input to align with the trained model's data structure.
    """
    # Convert user input to DataFrame
    input_df = pd.DataFrame([user_input])

    # One-hot encode categorical features
    input_encoded = pd.get_dummies(input_df, columns=cat_features)

    # Align the input features with the training dataset's columns
    missing_cols = set(train_columns) - set(input_encoded.columns)
    for col in missing_cols:
        input_encoded[col] = 0

    # Reorder columns to match training data
    input_encoded = input_encoded[train_columns]

    # Scale numerical features
    input_encoded[num_features] = scaler.transform(input_encoded[num_features])

    return input_encoded


@app.route("/", methods=["GET", "POST"])
def index():
    """
    Home page for the application.
    """
    if request.method == "POST":
        # Get form data
        user_input = {
            "age": int(request.form["age"]),
            "sex": request.form["sex"],
            "chest_pain_type": request.form["chest_pain_type"],
            "resting_blood_pressure": float(request.form["resting_blood_pressure"]),
            "cholestoral": float(request.form["cholestoral"]),
            "fasting_blood_sugar": request.form["fasting_blood_sugar"],
            "rest_ecg": request.form["rest_ecg"],
            "Max_heart_rate": float(request.form["Max_heart_rate"]),
            "exercise_induced_angina": request.form["exercise_induced_angina"],
            "oldpeak": float(request.form["oldpeak"]),
            "slope": request.form["slope"],
            "vessels_colored_by_flourosopy": int(request.form["vessels_colored_by_flourosopy"]),
            "thalassemia": request.form["thalassemia"],
        }

        # Preprocess the input
        input_encoded = preprocess_input(user_input)

        # Predict
        prob = rf_model.predict_proba(input_encoded)[0][1]
        classification = "Disease" if prob > 0.4 else "No Disease"

        return render_template(
            "index.html",
            prediction=f"Prediction: {classification} (Probability: {prob:.2%})",
        )

    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
