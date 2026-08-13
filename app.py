import os
import joblib
import pandas as pd
import numpy as np
from flask import Flask, request, render_template
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from datetime import datetime

app = Flask(__name__)

# -------------------- Load training data and fit encoders --------------------
try:
    df_train = pd.read_csv('salary_prediction_dataset_1500.csv')
    df_train["Certifications"] = df_train["Certifications"].fillna("None")
    df_train["Internship_Quality"] = df_train["Internship_Quality"].fillna("None")

    encoders = {}
    for col in ['Degree', 'College_Tier', 'Internship_Quality', 'Location']:
        le = LabelEncoder()
        le.fit(df_train[col])
        encoders[col] = le

    mlb_skills = joblib.load('skills_encoder.pkl')
    mlb_cert = joblib.load('cert_encoder.pkl')
    feature_columns = joblib.load('feature_columns.pkl')
    model = joblib.load('salary_model_regressor.pkl')

except FileNotFoundError as e:
    print("ERROR: Missing required files. Make sure the CSV and all .pkl files are in the same directory.")
    raise e

# -------------------- Flask routes --------------------
@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    error = None
    
    # Default sample values
    sample = {
        'degree': 'B.Tech',
        'college_tier': 'Tier 1',
        'skills': ['Machine Learning', 'Python', 'SQL'],
        'certifications': ['AWS Solutions Architect'],
        'internship_quality': 'Good',
        'location': 'Bangalore',
        'experience_years': 5.0,
        'projects': 4
    }

    # Variables for report
    degree = sample['degree']
    college_tier = sample['college_tier']
    skills_list = sample['skills']
    certs_list = sample['certifications']
    internship_quality = sample['internship_quality']
    location = sample['location']
    experience_years = sample['experience_years']
    projects = sample['projects']

    if request.method == 'POST':
        try:
            # Get form data
            degree = request.form['degree']
            college_tier = request.form['college_tier']
            skills_list = request.form.getlist('skills')
            certs_list = request.form.getlist('certifications')
            internship_quality = request.form['internship_quality']
            location = request.form['location']
            experience_years = float(request.form['experience_years'])
            projects = int(request.form['projects'])

            # Transform categoricals
            degree_enc = encoders['Degree'].transform([degree])[0]
            college_enc = encoders['College_Tier'].transform([college_tier])[0]
            internship_enc = encoders['Internship_Quality'].transform([internship_quality])[0]
            location_enc = encoders['Location'].transform([location])[0]

            # Binarize skills and certifications
            skills_encoded = mlb_skills.transform([skills_list])
            certs_encoded = mlb_cert.transform([certs_list])

            # Build input dictionary
            input_dict = {col: 0 for col in feature_columns}

            input_dict['Degree'] = degree_enc
            input_dict['College_Tier'] = college_enc
            input_dict['Internship_Quality'] = internship_enc
            input_dict['Location'] = location_enc
            input_dict['Experience_Years'] = experience_years
            input_dict['Projects'] = projects

            for i, skill in enumerate(mlb_skills.classes_):
                if skill in feature_columns:
                    input_dict[skill] = skills_encoded[0, i]

            for i, cert in enumerate(mlb_cert.classes_):
                if cert in feature_columns:
                    input_dict[cert] = certs_encoded[0, i]

            input_df = pd.DataFrame([input_dict])[feature_columns]

            # Predict
            pred = model.predict(input_df)[0]
            prediction = round(pred, 2)
            
            # Reset sample values after successful prediction
            sample = {
                'degree': 'B.Tech',
                'college_tier': 'Tier 1',
                'skills': [],
                'certifications': [],
                'internship_quality': 'Good',
                'location': 'Bangalore',
                'experience_years': 0.0,
                'projects': 0
            }

        except Exception as e:
            error = str(e)

    # Get current time for report
    now = datetime.now()
    
    return render_template('index.html',
                           prediction=prediction,
                           error=error,
                           sample=sample,
                           degree=degree,
                           college_tier=college_tier,
                           skills_list=skills_list,
                           certs_list=certs_list,
                           internship_quality=internship_quality,
                           location=location,
                           experience_years=experience_years,
                           projects=projects,
                           now=now)

if __name__ == '__main__':
    app.run(debug=True)
