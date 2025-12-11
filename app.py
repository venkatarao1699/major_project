import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from flask import Flask, render_template, request
import os

app = Flask(__name__)

# -----------------------------------------------------------
# LOAD & TRAIN MODEL ONCE (Railway Safe)
# -----------------------------------------------------------

df = pd.read_csv("stress_detection_IT_professionals_dataset.csv")

# Correct columns:
# ['Heart_Rate', 'Skin_Conductivity', 'Hours_Worked', 'Stress_Level', 'Emails_Sent', 'Meetings_Attended']

X = df.drop("Stress_Level", axis=1)
y = df["Stress_Level"]

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor()
model.fit(x_train, y_train)

print("Model trained successfully using uploaded dataset!")


# -----------------------------------------------------------
# ROUTES
# -----------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/prediction", methods=["GET", "POST"])
def prediction():
    if request.method == "POST":
        # Collect inputs (names must match your HTML form names)
        heart = float(request.form["heart"])
        skin = float(request.form["skin"])
        hours = float(request.form["hours"])
        emails = float(request.form["emails"])
        meetings = float(request.form["meetings"])

        # Prepare input
        input_data = np.array([[heart, skin, hours, emails, meetings]])

        # Reorder to match X column order
        # Columns: Heart_Rate, Skin_Conductivity, Hours_Worked, Emails_Sent, Meetings_Attended
        input_data = np.array([[heart, skin, hours, emails, meetings]])

        # Prediction
        stress_pred = model.predict(input_data)[0]

        # Category logic
        if stress_pred <= 30:
            category = "Low Stress 😌"
        elif stress_pred <= 50:
            category = "Moderate Stress 🙂"
        elif stress_pred <= 70:
            category = "High Stress 😥"
        else:
            category = "Severe Stress 🚨"

        msg = f"Predicted Stress Level: {stress_pred:.2f} • {category}"

        return render_template("prediction.html", msg=msg)

    return render_template("prediction.html")


# -----------------------------------------------------------
# RUN ON RAILWAY PORT
# -----------------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
