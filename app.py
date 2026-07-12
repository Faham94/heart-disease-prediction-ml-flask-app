from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import pandas as pd
import os

app = Flask(__name__)

# Load model artifacts
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model   = joblib.load(os.path.join(BASE_DIR, 'SVM_heart.pkl'))
scalar  = joblib.load(os.path.join(BASE_DIR, 'scalar.pkl'))
columns = joblib.load(os.path.join(BASE_DIR, 'columns.pkl'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        # Build a one-row DataFrame with the same columns the model was trained on
        row = {col: 0 for col in columns}

        # Numeric fields
        row['Age']            = float(data['Age'])
        row['RestingBP']      = float(data['RestingBP'])
        row['Cholesterol']    = float(data['Cholesterol'])
        row['FastingBS']      = int(data['FastingBS'])
        row['MaxHR']          = float(data['MaxHR'])
        row['Oldpeak']        = float(data['Oldpeak'])

        # One-hot encoded fields (drop_first=True was used so first category is baseline)
        sex = data.get('Sex', 'M')
        if sex == 'M':
            row['Sex_M'] = 1

        cpt = data.get('ChestPainType', 'ATA')
        for val in ['ATA', 'NAP', 'TA']:
            key = f'ChestPainType_{val}'
            if key in row:
                row[key] = 1 if cpt == val else 0

        recg = data.get('RestingECG', 'Normal')
        for val in ['Normal', 'ST']:
            key = f'RestingECG_{val}'
            if key in row:
                row[key] = 1 if recg == val else 0

        ea = data.get('ExerciseAngina', 'N')
        if 'ExerciseAngina_Y' in row:
            row['ExerciseAngina_Y'] = 1 if ea == 'Y' else 0

        slope = data.get('ST_Slope', 'Up')
        for val in ['Flat', 'Up']:
            key = f'ST_Slope_{val}'
            if key in row:
                row[key] = 1 if slope == val else 0

        df_input = pd.DataFrame([row])[columns]
        scaled   = scalar.transform(df_input)
        pred     = model.predict(scaled)[0]
        prob     = model.decision_function(scaled)[0]

        # Convert decision function to a 0-1 "confidence" proxy via sigmoid
        confidence = round(1 / (1 + np.exp(-abs(prob))) * 100, 1)

        return jsonify({
            'prediction': int(pred),
            'label': 'Heart Disease Detected' if pred == 1 else 'No Heart Disease',
            'confidence': confidence
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
