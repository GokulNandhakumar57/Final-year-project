import pandas as pd
import lightgbm as lgb
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score

# Load the dataset
df = pd.read_csv("diabetes_prediction_dataset.csv")

# Fill missing values if any
df.fillna(method="ffill", inplace=True)

# Encode categorical columns
le_gender = LabelEncoder()
le_smoking = LabelEncoder()
df["gender"] = le_gender.fit_transform(df["gender"])
df["smoking_history"] = le_smoking.fit_transform(df["smoking_history"])

# Features and target
features = ["gender", "age", "hypertension", "heart_disease", "smoking_history",
            "bmi", "HbA1c_level", "blood_glucose_level"]
X = df[features]
y = df["diabetes"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define and train the model
model = lgb.LGBMClassifier(objective="binary", boosting_type="gbdt", learning_rate=0.05, n_estimators=100)
model.fit(X_train, y_train, eval_set=[(X_test, y_test)], callbacks=[lgb.early_stopping(10)])

# Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Save the model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

# Save encoders
with open("encoders.pkl", "wb") as f:
    pickle.dump({"gender": le_gender, "smoking_history": le_smoking}, f)
