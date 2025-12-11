import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.tree import ExtraTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from flask import Flask, render_template, request
import os

app = Flask(__name__)

# -----------------------------------------------------------
# 🔥 TRAIN MODEL ONCE WHEN THE APP STARTS
# -----------------------------------------------------------

df = pd.read_csv("stress_detection_IT_professionals_dataset.csv")

# Preprocess dataset
x = df.drop("Stress Level", axis=1)
y = df["Stress Level"]

# Split dataset
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

# Use a model (RandomForest is good for this)
model = RandomForestRegressor()
model.fit(x_train, y_train)

print("Model trained successfully on startup!")


# -----------------------------------------------------------
# ROUTES
# -----------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/prediction", methods=["GET", "POST"])
def prediction():
    if request.method == "POST":

        # Collect form inputs
        age = float(request.form["age"])
        work_hours = float(request.form["work_hours"])
        sleep_hours = float(request.form["sleep_hours"])
        anxiety = float(request.form["anxiety"])
        workload = float(request.form["workload"])

        # Prepare input features
        input_features = np.array([[age, work_hours, sleep_hours, anxiety, workload]])

        # Predict
        stress_level = model.predict(input_features)[0]

        # Categorize stress level
        if stress_level <= 40:
            category = "Low Stress - Stay hydrated and take short breaks. 🍵"
        elif stress_level <= 60:
            category = "Moderate Stress - Try relaxation exercises. 🧘"
        elif stress_level <= 80:
            category = "High Stress - Slow down and manage workload. ⚠️"
        else:
            category = "Severe Stress - Take immediate action; seek support. 🚨"

        msg = f"The predicted stress level is {stress_level:.2f}%. {category}"

        return render_template("prediction.html", msg=msg)

    return render_template("prediction.html")


# -----------------------------------------------------------
# 🔥 RUN APP ON RAILWAY PORT
# -----------------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
