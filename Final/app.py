from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import os

app = Flask(__name__, static_folder='static', template_folder='templates')

# Load model and scaler using relative paths
model = joblib.load(os.path.join("model", "random_forest_model.pkl"))
scaler = joblib.load(os.path.join("model", "scaler.pkl"))

# Mappings
occupation_mapping = {
    'Software Engineer': 0, 'Doctor': 1, 'Sales Representative': 2, 'Teacher': 3,
    'Nurse': 4, 'Engineer': 5, 'Accountant': 6, 'Scientist': 7, 'Lawyer': 8,
    'Salesperson': 9, 'Manager': 10, 'Student': 11, 'Athlete': 12, 'Artist': 13
}
gender_mapping = {'Female': 0, 'Male': 1}

# Helper functions
def calculate_bmi(weight, height_cm):
    height_m = height_cm / 100
    return weight / (height_m ** 2)

def get_bmi_category(bmi):
    if bmi < 18.5:
        return "Underweight"
    elif 18.5 <= bmi < 25:
        return "Normal"
    else:
        return "Overweight"

def predict_sleep_disorder(score, stress, duration):
    if score < 4 or (stress > 7 and duration < 5):
        return "🚨 High risk of sleep disorder. Consult a specialist."
    elif score < 6:
        return "⚠️ Moderate risk. Consider improving sleep hygiene."
    return "✅ Low risk. Maintain healthy sleep habits."

# Routes
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    try:
        age = int(data["age"])
        gender = data["gender"]
        occupation = data["occupation"]
        sleep_duration = float(data["sleepDuration"])
        activity = int(data["activityLevel"])
        stress = int(data["stressLevel"])
        steps = int(data["steps"])
        weight = float(data["weight"])
        height = float(data["height"])
        systolic = int(data["systolic"])
        diastolic = int(data["diastolic"])
        heart_rate = int(data["heartRate"])

        bmi = calculate_bmi(weight, height)
        bmi_category = get_bmi_category(bmi)

        user_df = pd.DataFrame([{
            'Gender': gender_mapping[gender],
            'Age': age,
            'Occupation': occupation_mapping[occupation],
            'Sleep Duration': sleep_duration,
            'Physical Activity Level': activity,
            'Stress Level': stress,
            'Systolic BP': systolic,
            'Diastolic BP': diastolic,
            'Heart Rate': heart_rate,
            'Daily Steps': steps
        }])

        scaled_input = scaler.transform(user_df)
        sleep_score = model.predict(scaled_input)[0]
        disorder_risk = predict_sleep_disorder(sleep_score, stress, sleep_duration)

        return jsonify({
            "bmi": round(bmi, 2),
            "bmi_category": bmi_category,
            "sleep_score": round(sleep_score, 1),
            "disorder_risk": disorder_risk
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400

# Run app
if __name__ == "__main__":
    app.run(debug=True)

import webbrowser
import threading

def open_browser():
    webbrowser.open_new("http://127.0.0.1:5000")

if __name__ == '__main__':
    threading.Timer(1.25, open_browser).start()
    app.run(debug=True)

