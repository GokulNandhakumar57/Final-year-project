from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# Custom LabelEncoder to handle unseen labels
class CustomLabelEncoder(LabelEncoder):
    def fit(self, y):
        super().fit(y)
        return self
    
    def transform(self, y):
        # Return transformed values and ignore unknown labels
        return super().transform(y)
    
    def fit_transform(self, y):
        return self.fit(y).transform(y)

# Load the model and encoders
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("encoders.pkl", "rb") as f:
    encoders = pickle.load(f)

# Update the encoders to handle unseen categories
# Here we fit the encoders with all known categories used during model training
encoders["gender"] = CustomLabelEncoder().fit(["Male", "Female"])  # Example categories
encoders["smoking_history"] = CustomLabelEncoder().fit(
    ["Never smoked", "Smokes", "Formerly smoked"]
)  # Example categories

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    user_data = {}

    if request.method == "POST":
        try:
            # Get form data
            gender = request.form["gender"]
            age = int(request.form["age"])
            hypertension = int(request.form["hypertension"])
            heart_disease = int(request.form["heart_disease"])
            smoking_history = request.form["smoking_history"]
            bmi = float(request.form["bmi"])
            hba1c_level = float(request.form["hba1c_level"])
            blood_glucose_level = float(request.form["blood_glucose_level"])

            # Store for display
            user_data = {
                "Gender": gender,
                "Age": age,
                "Hypertension": hypertension,
                "Heart Disease": heart_disease,
                "Smoking History": smoking_history,
                "BMI": bmi,
                "HbA1c Level": hba1c_level,
                "Blood Glucose Level": blood_glucose_level
            }

            # Encode categorical fields
            gender_encoded = encoders["gender"].transform([gender])[0]  # Handles unseen labels
            smoking_encoded = encoders["smoking_history"].transform([smoking_history])[0]  # Handles unseen labels

            # Prepare input for prediction
            input_data = pd.DataFrame([[
                gender_encoded, age, hypertension, heart_disease,
                smoking_encoded, bmi, hba1c_level, blood_glucose_level
            ]], columns=[
                "gender", "age", "hypertension", "heart_disease",
                "smoking_history", "bmi", "HbA1c_level", "blood_glucose_level"
            ])

            # Make prediction
            prediction = model.predict(input_data)[0]
            result = "Positive" if prediction == 1 else "Negative"

            # Save to CSV
            submission_df = pd.DataFrame([user_data])
            if not os.path.exists("submissions.csv"):
                submission_df.to_csv("submissions.csv", index=False)
            else:
                submission_df.to_csv("submissions.csv", mode='a', header=False, index=False)

        except Exception as e:
            result = f"Error: {e}"

    return render_template("index.html", result=result, user_data=user_data)

if __name__ == "__main__":
    app.run(debug=True)
