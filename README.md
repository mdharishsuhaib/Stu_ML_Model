# Student Performance Prediction Model (Stu_ML_Model)

An end-to-end Machine Learning project to predict student exam scores (`Exam_Score`) based on academic, behavioral, and demographic indicators.

## 🚀 Project Overview
This repository contains a high-accuracy, hyperparameter-tuned **HistGradientBoostingRegressor** machine learning pipeline trained on actual student demographics and habits data. This model has been enhanced using automated feature engineering to maximize predictive performance ($R^2 \approx 73.99\%$ with a low Mean Absolute Error of $\approx 0.75$ exam points).

## 🛠️ Features Engineered
- **Study Efficiency Index**: Ratio of study hours to sleep hours.
- **Total Support Score**: Composite score combining Parental Involvement, Access to Resources, and Teacher Quality.
- **Academic Engagement**: Interaction multiplier of study hours and tutoring sessions.

## 📦 Workspace Assets & Directory Structure
- `StudentPerformanceFactors.csv`: Dataset used for training and validation.
- `final_student_performance_model.joblib`: The serialized, full-stage prediction pipeline containing preprocessing (imputers, standardizers, encoders) and the trained gradient boosting model.
- `app.py`: A lightweight, production-ready Flask API service that accepts JSON payload configurations, dynamically engineers the input features, and returns predictions instantly.
- `requirements.txt`: Runtime dependencies, including the scikit-learn version used to serialize the model.

## ⚙️ Installation & Local Setup

1. Clone the repository:
```bash
git clone https://github.com/mdharishsuhaib/Stu_ML_Model.git
cd Stu_ML_Model
```

2. Install matching dependencies:
```bash
pip install -r requirements.txt
```

3. Run the Flask Web API:
```bash
python app.py
```

The API runs at `http://127.0.0.1:5001`. Check that the model loaded successfully:

```bash
curl http://127.0.0.1:5001/health
```

To test a row from the dataset on Windows Command Prompt, use:

```cmd
python -c "import pandas as pd,requests,json; d=pd.read_csv('StudentPerformanceFactors.csv').dropna(); row=d.iloc[0]; payload=json.loads(row.drop('Exam_Score').to_json()); r=requests.post('http://127.0.0.1:5001/predict',json=payload); print('Actual Exam_Score:',row['Exam_Score']); print('Prediction:',r.text)"
```

## 📡 API Usage & Live Testing

Send an HTTP POST request to the `/predict` endpoint (running on port `5001` or `5002`): 

**Endpoint:** `POST http://127.0.0.1:5001/predict`

**Payload Format:**
```json
{
  "Hours_Studied": 18,
  "Attendance": 98,
  "Parental_Involvement": "High",
  "Access_to_Resources": "High",
  "Extracurricular_Activities": "Yes",
  "Sleep_Hours": 8,
  "Previous_Scores": 90,
  "Motivation_Level": "High",
  "Internet_Access": "Yes",
  "Tutoring_Sessions": 3,
  "Family_Income": "High",
  "Teacher_Quality": "High",
  "School_Type": "Private",
  "Peer_Influence": "Positive",
  "Physical_Activity": 4,
  "Learning_Disabilities": "No",
  "Parental_Education_Level": "Postgraduate",
  "Distance_from_Home": "Near",
  "Gender": "Female"
}
```

**Expected Response:**
```json
{
  "predictions": [77.74]
}
```
