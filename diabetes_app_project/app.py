from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64

app = Flask(__name__)

# Load model
model = pickle.load(open("model.pkl", "rb"))

# Encode mappings
gender_map = {"Male": 1, "Female": 0, "Other": 2}
smoking_map = {"never": 0, "former": 1, "current": 2, "ever": 3, "not current": 4}

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    if request.method == "POST":
        try:
            gender = gender_map[request.form["gender"]]
            age = int(request.form["age"])
            hypertension = int(request.form["hypertension"])
            heart_disease = int(request.form["heart_disease"])
            smoking = smoking_map[request.form["smoking_history"]]
            bmi = float(request.form["bmi"])
            hba1c = float(request.form["HbA1c_level"])
            glucose = float(request.form["blood_glucose_level"])

            features = np.array([[gender, age, hypertension, heart_disease, smoking, bmi, hba1c, glucose]])
            prediction = model.predict(features)[0]

            # Save the submission
            new_row = pd.DataFrame([{
                "gender": request.form["gender"],
                "age": age,
                "hypertension": hypertension,
                "heart_disease": heart_disease,
                "smoking_history": request.form["smoking_history"],
                "bmi": bmi,
                "HbA1c_level": hba1c,
                "blood_glucose_level": glucose,
                "diabetes": prediction
            }])
            new_row.to_csv("submissions.csv", mode="a", header=False, index=False)

            result = "Diabetic" if prediction == 1 else "Non-Diabetic"

        except Exception as e:
            result = f"Error in prediction: {e}"

    return render_template("index.html", result=result)

@app.route("/analysis")
def analysis():
    try:
        df = pd.read_csv("diabetes_prediction_dataset.csv")

        features = [
            "gender", "age", "hypertension", "heart_disease", "smoking_history",
            "bmi", "HbA1c_level", "blood_glucose_level", "diabetes"
        ]

        plots = []
        for col in features[:-1]:  # skip 'diabetes'
            plt.figure(figsize=(6, 4))
            if df[col].dtype == "object":
                sns.countplot(data=df, x=col, hue="diabetes")
            else:
                sns.boxplot(data=df, x="diabetes", y=col)
            plt.title(f"{col} vs Diabetes")
            plt.tight_layout()

            buf = io.BytesIO()
            plt.savefig(buf, format="png")
            buf.seek(0)
            image_base64 = base64.b64encode(buf.read()).decode("utf-8")
            buf.close()
            plt.close()
            plots.append(image_base64)

        return render_template("analysis.html", plots=plots)

    except Exception as e:
        return f"Error loading analysis: {e}"

if __name__ == "__main__":
    app.run(debug=True)
