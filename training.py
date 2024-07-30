import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib

def load_data(symptom_file, medication_file):
    symptom_data = pd.read_csv(symptom_file)
    medication_data = pd.read_csv(medication_file)
    return symptom_data, medication_data

def preprocess_data(symptom_data):
    symptom_cols = ['Symptom_1', 'Symptom_2', 'Symptom_3', 'Symptom_4']
    X = symptom_data[symptom_cols].astype(str).apply(lambda x: x.str.lower().str.strip(), axis=1).values.tolist()
    X = [list(set(symptoms)) for symptoms in X] 
    y = symptom_data['Disease']

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    mlb = MultiLabelBinarizer()
    X_encoded = mlb.fit_transform(X)
    
    return X_encoded, y_encoded, label_encoder, mlb

def train_model(symptom_file, medication_file):
    symptom_data, medication_data = load_data(symptom_file, medication_file)
    X_encoded, y_encoded, label_encoder, mlb = preprocess_data(symptom_data)
    
    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y_encoded, test_size=0.3, random_state=42)
    
    model = SVC(kernel='linear', probability=True)
    model.fit(X_train, y_train)
    
    y_predicted = model.predict(X_test)
    model_accuracy = accuracy_score(y_test, y_predicted)
    model_precision = precision_score(y_test, y_predicted, average='weighted')
    model_recall = recall_score(y_test, y_predicted, average='weighted')
    model_f1_score = f1_score(y_test, y_predicted, average='weighted')
    model_roc_auc = roc_auc_score(y_test, model.predict_proba(X_test), multi_class='ovr')

    print(f'Best Model Accuracy: {model_accuracy}')
    print(f'Precision: {model_precision}')
    print(f'Recall: {model_recall}')
    print(f'F1 Score: {model_f1_score}')
    print(f'ROC AUC: {model_roc_auc}')
    
    joblib.dump(model, 'disease_prediction_model.pkl')
    joblib.dump(label_encoder, 'label_encoder.pkl')
    joblib.dump(mlb, 'mlb.pkl')

if __name__ == "__main__":

    train_model('data/symptoms.csv', 'data/medications.csv')
