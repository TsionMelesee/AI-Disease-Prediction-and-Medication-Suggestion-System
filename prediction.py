import numpy as np
import joblib

from training import load_data

symptom_data, medication_data = load_data('data/symptoms_df.csv', 'data/medications.csv')

model = joblib.load('disease_prediction_model.pkl')
label_encoder = joblib.load('label_encoder.pkl')
mlb = joblib.load('mlb.pkl')

def validate_symptoms(symptoms, mlb):
    categories = mlb.classes_
    validated_symptoms = [symptom if symptom in categories else 'unknown' for symptom in symptoms]
    return validated_symptoms

def predict_disease_medication(symptoms):
    symptoms = validate_symptoms(symptoms, mlb)
    
    symptoms_df = [list(set(symptoms))]
    
    symptoms_encoded = mlb.transform(symptoms_df)
    
    disease_probabilities = model.predict_proba(symptoms_encoded)[0]
    
    sorted_disease_indices = np.argsort(disease_probabilities)[::-1]
    predicted_disease_index = sorted_disease_indices[0]
    predicted_disease = label_encoder.inverse_transform([predicted_disease_index])[0]
    
    try:
        medication = medication_data.loc[medication_data['Disease'] == predicted_disease, 'Medication'].values[0]
    except IndexError:
        medication = "No medication found for the predicted disease."
    
    return predicted_disease, medication

symptoms = ['watering_from_eyes', 'chills', 'continuous_sneezing', 'shivering', 'watering_from_eyes', 'watering_from_eyes']
predicted_disease, medication = predict_disease_medication(symptoms)
print(f'Disease: {predicted_disease}')
print(f'Medication: {medication}')
