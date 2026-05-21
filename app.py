import streamlit as st
import joblib
import numpy as np
import os

st.set_page_config(page_title="Student Score Predictor", page_icon="🎓")

MODEL_PATH = "model/model.pkl"

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error("Model not found. Run `python train.py` first.")
        st.stop()
    return joblib.load(MODEL_PATH)

model = load_model()

# ---- UI ----
st.title("🎓 Student Score Predictor")
st.markdown("Predict a student's final score based on study habits.")

st.divider()

col1, col2, col3 = st.columns(3)

with col1:
    study_hours = st.slider("📚 Study Hours / day", 0.0, 10.0, 5.0, 0.5)

with col2:
    attendance = st.slider("🏫 Attendance (%)", 50, 100, 80)

with col3:
    assignments = st.slider("📝 Assignments (%)", 0, 100, 75)

st.divider()

if st.button("Predict Score", use_container_width=True, type="primary"):
    features = np.array([[study_hours, attendance, assignments]])
    score    = float(model.predict(features)[0])
    score    = round(min(max(score, 0), 100), 1)

    if score >= 90:
        grade, color = "A", "green"
    elif score >= 80:
        grade, color = "B", "blue"
    elif score >= 70:
        grade, color = "C", "orange"
    elif score >= 60:
        grade, color = "D", "red"
    else:
        grade, color = "F", "red"

    st.metric("Predicted Score", f"{score} / 100")
    st.markdown(f"**Grade:** :{color}[{grade}]")

    if score >= 70:
        st.success("Great performance! Keep it up.")
    elif score >= 60:
        st.warning("Passing, but there's room to improve.")
    else:
        st.error("At risk. Consider increasing study hours and attendance.")

st.divider()
st.caption("Model: Linear Regression  •  Features: study hours, attendance, assignments")