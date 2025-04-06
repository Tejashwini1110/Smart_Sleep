import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

# Load trained model and scaler
model = joblib.load("random_forest_model.pkl")
scaler = joblib.load("scaler.pkl")

# Define mappings for categorical inputs
occupation_mapping = {
    'Software Engineer': 0, 'Doctor': 1, 'Sales Representative': 2, 'Teacher': 3,
    'Nurse': 4, 'Engineer': 5, 'Accountant': 6, 'Scientist': 7, 'Lawyer': 8,
    'Salesperson': 9, 'Manager': 10, 'Student': 11, 'Athlete': 12, 'Artist': 13
}
gender_mapping = {'Female': 0, 'Male': 1}

# BMI Calculation
def calculate_bmi(weight, height):
    return weight / (height ** 2)

def get_bmi_category(bmi):
    if bmi < 18.5:
        return "Underweight"
    elif 18.5 <= bmi < 25:
        return "Normal"
    else:
        return "Overweight"

# Sleep Disorder Prediction
def predict_sleep_disorder(quality_of_sleep, stress_level, sleep_duration):
    if quality_of_sleep < 4 or (stress_level > 7 and sleep_duration < 5):
        return "🚨 High risk of sleep disorder. Consult a specialist."
    elif quality_of_sleep < 6:
        return "⚠️ Moderate risk. Consider improving sleep hygiene."
    else:
        return "✅ Low risk. Maintain healthy sleep habits."

# Sleep Report & Recommendations
def generate_sleep_report(quality_of_sleep, sleep_duration, stress_level, physical_activity, 
                          age, occupation, bmi_category, heart_rate, systolic_bp, diastolic_bp):
    report = f"\n🔹 **Predicted Quality of Sleep: {quality_of_sleep:.1f}/10**\n\n"
    report += "### 🌜 Sleep Analysis Report\n"

    # Sleep duration analysis
    report += f"- **Sleep Duration:** {sleep_duration} hours "
    if sleep_duration < 6:
        report += "(⚠️ Below recommended 7-9 hours)\n"
    elif sleep_duration > 9:
        report += "(⚠️ Above recommended amount)\n"
    else:
        report += "(✅ Optimal duration)\n"
    
    # Sleep quality assessment
    report += "\n### 🛌 Quality Assessment\n"
    if quality_of_sleep >= 8:
        report += "- 🌟 Excellent! You're getting restorative sleep\n"
    elif quality_of_sleep >= 6:
        report += "- 👍 Good, but room for improvement\n"
    else:
        report += "- 😟 Needs improvement for better health\n"
    
    # Personalized recommendations section
    report += "\n### 💡 Personalized Recommendations\n"

    # Sleep duration improvement
    if sleep_duration < 6:
        report += "- 🕘 **Go to bed earlier**: Aim for 7-9 hours nightly\n"
        if occupation in [11, 5]:  # Student or Engineer
            report += "  - As a busy professional, try power naps (20-30 mins) to supplement\n"
    
    # Stress management
    if stress_level > 6:
        report += "- 🧘 **Stress reduction**: Try these before bed:\n"
        report += "  - 4-7-8 breathing technique (inhale 4s, hold 7s, exhale 8s)\n"
        report += "  - Progressive muscle relaxation\n"
        if age > 50:
            report += "  - Gentle yoga or tai chi can be especially helpful\n"
    
    # Physical activity recommendations
    if physical_activity < 4:
        report += "- 🏃 **Move more**: Aim for at least 30 mins daily activity\n"
        report += "  - Morning walks help regulate circadian rhythm\n"
    elif physical_activity > 7:
        report += "- ⏰ **Timing matters**: Avoid intense workouts within 3 hours of bedtime\n"
    
    # Occupation-based suggestions
    if occupation in [1, 4]:  # Doctor or Nurse
        report += "- ⚕️ **For healthcare workers**:\n"
        report += "  - Maintain consistent sleep schedule even on days off\n"
        report += "  - Use blackout curtains for daytime sleeping\n"
    
    # BMI-related suggestions
    if bmi_category in ["Overweight"]:
        report += "- 🍏 **Weight management**:\n"
        report += "  - Avoid heavy meals before bedtime\n"
        report += "  - Consider sleep apnea screening if you snore\n"
    
    # Heart rate recommendations
    if heart_rate > 80:
        report += "- ❤️ **Heart health**:\n"
        report += "  - Evening meditation may help lower resting heart rate\n"

    # Blood pressure considerations
    if systolic_bp > 130 or diastolic_bp > 85:
        report += "- 💓 **Blood Pressure Alert:**\n"
        report += "  - High BP can disrupt sleep. Reduce salt intake and manage stress.\n"

    # General sleep hygiene
    report += "\n### 🌙 Universal Sleep Tips\n"
    report += "- 📵 Create a tech-free zone 1 hour before bed\n"
    report += "- 🌡️ Keep bedroom temperature between 60-67°F (15-19°C)\n"
    report += "- 🛏️ Reserve bed only for sleep (no work or TV)\n"

    # Morning routine suggestions
    report += "\n### ☀️ Morning Boosters\n"
    report += "- Open curtains immediately upon waking\n"
    if quality_of_sleep < 6:
        report += "- Consider a dawn simulator alarm clock\n"

    # Final encouragement
    report += "\n💬 Remember: Small consistent changes make the biggest difference!"
    
    return report

# Streamlit UI
st.title("💤 Smart Sleep Quality Predictor")

# User Inputs
gender = st.selectbox("Gender", ["Female", "Male"])
age = st.number_input("Age", min_value=10, max_value=100, value=25)
occupation = st.selectbox("Occupation", list(occupation_mapping.keys()))
sleep_duration = st.number_input("Sleep Duration (hours)", min_value=0.0, max_value=12.0, value=7.0)
physical_activity = st.slider("Physical Activity Level (0-10)", 0, 10, 5)
stress_level = st.slider("Stress Level (1-10)", 1, 10, 5)
weight = st.number_input("Weight (kg)", min_value=30.0, max_value=150.0, value=70.0)
height = st.number_input("Height (cm)", min_value=100.0, max_value=220.0, value=170.0) / 100  # Convert to meters
systolic_bp = st.number_input("Systolic BP", min_value=80, max_value=180, value=120)
diastolic_bp = st.number_input("Diastolic BP", min_value=50, max_value=120, value=80)
heart_rate = st.number_input("Heart Rate (BPM)", min_value=40, max_value=120, value=70)
daily_steps = st.number_input("Daily Steps", min_value=0, max_value=30000, value=5000)

# Calculate BMI
bmi = calculate_bmi(weight, height)
bmi_category = get_bmi_category(bmi)

# Display BMI
st.subheader("📊 Your BMI Results")
st.write(f"**BMI Value:** {bmi:.2f}")
st.write(f"**Category:** {bmi_category}")

# Prediction Button
if st.button("Predict Sleep Quality"):
    user_data = pd.DataFrame([{
        'Gender': gender_mapping[gender], 'Age': age, 'Occupation': occupation_mapping[occupation],
        'Sleep Duration': sleep_duration, 'Physical Activity Level': physical_activity,
        'Stress Level': stress_level, 'Systolic BP': systolic_bp, 'Diastolic BP': diastolic_bp,
        'Heart Rate': heart_rate, 'Daily Steps': daily_steps
    }])
    
    user_data_scaled = scaler.transform(user_data)
    predicted_quality = model.predict(user_data_scaled)[0]

    # Display Predictions
    st.subheader("📈 Sleep Quality Prediction")
    st.write(f"**Predicted Sleep Quality:** {predicted_quality:.1f}/10")

    # Sleep Disorder Risk
    disorder_risk = predict_sleep_disorder(predicted_quality, stress_level, sleep_duration)
    st.subheader("🚨 Sleep Disorder Risk")
    st.write(disorder_risk)

    # Sleep Report
    sleep_report = generate_sleep_report(predicted_quality, sleep_duration, stress_level, physical_activity, age, occupation_mapping[occupation], bmi_category, heart_rate, systolic_bp, diastolic_bp)
    
    st.markdown(sleep_report)

