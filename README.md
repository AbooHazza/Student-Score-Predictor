# Student Score Predictor 

Predict a student's final score based on:
- **Study Hours** per day
- **Attendance** percentage
- **Assignments** completion percentage

## Project Structure

```
student-score-predictor/
├── data/               # Auto-generated dataset
├── model/              # Saved model file
├── app.py              # Streamlit web app
├── train.py            # Data generation + model training
├── requirements.txt
└── README.md
```

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train the model (generates data + saves model)
python train.py

# 3. Run the app
py -m streamlit run app.py
```

## Model

Linear Regression trained on 300 synthetic student records.

Score formula (approximate):
```
score = study_hours × 4.0 + attendance × 0.3 + assignments × 0.2 + noise
```

| Feature       | Weight |
|---------------|--------|
| Study Hours   | 4.0    |
| Attendance    | 0.3    |
| Assignments   | 0.2    |

## Grades

| Score    | Grade |
|----------|-------|
| 90 – 100 | A     |
| 80 – 89  | B     |
| 70 – 79  | C     |
| 60 – 69  | D     |
| 0  – 59  | F     |
