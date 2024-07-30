from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

disease_model = joblib.load('disease_prediction_model.pkl')
label_encoder = joblib.load('label_encoder.pkl')
mlb = joblib.load('mlb.pkl')
symptom_model = joblib.load('symptom_suggestion_model.pkl')
symptom_vectorizer = joblib.load('symptom_vectorizer.pkl')

medication_data = pd.read_csv('data/medications.csv')

def validate_symptoms(symptoms, mlb):
    categories = mlb.classes_
    validated_symptoms = [symptom if symptom in categories else 'unknown' for symptom in symptoms]
    return validated_symptoms

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/suggest_symptoms', methods=['POST'])
def suggest_symptoms():
    data = request.get_json()
    partial_symptom = data.get('partial_symptom', '')

    if not partial_symptom:
        return jsonify([])

    partial_symptom_vect = symptom_vectorizer.transform([partial_symptom])
    probas = symptom_model.predict_proba(partial_symptom_vect)[0]
    top_indices = np.argsort(probas)[-5:][::-1]
    suggested_symptoms = [symptom_model.classes_[i] for i in top_indices]

    return jsonify(suggested_symptoms)


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    symptoms = data.get('symptoms')


    symptoms = validate_symptoms(symptoms, mlb)
    symptoms_df = [list(set(symptoms))]
    symptoms_encoded = mlb.transform(symptoms_df)
    
    disease_probabilities = disease_model.predict_proba(symptoms_encoded)[0]
    sorted_disease_indices = np.argsort(disease_probabilities)[::-1]
    predicted_disease_index = sorted_disease_indices[0]
    predicted_disease = label_encoder.inverse_transform([predicted_disease_index])[0]
    
    try:
        medication = medication_data.loc[medication_data['Disease'] == predicted_disease, 'Medication'].values[0]
    except IndexError:
        medication = "No medication found for the predicted disease."
    
    return jsonify({
        'disease': predicted_disease,
        'medication': medication
    })

if __name__ == '__main__':
    app.run(debug=True)
