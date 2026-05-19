import pandas as pd
import mlflow
import dagshub
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

mlflow.set_tracking_uri("mlruns")

# Load Dataset Hasil Preprocessing
current_dir = os.path.dirname(os.path.abspath(__file__))
dataset_dir = os.path.join(current_dir, "wine_quality_preprocessing")

X_train = pd.read_csv(os.path.join(dataset_dir, "X_train.csv"))
y_train = pd.read_csv(os.path.join(dataset_dir, "y_train.csv")).values.ravel()
X_test = pd.read_csv(os.path.join(dataset_dir, "X_test.csv"))
y_test = pd.read_csv(os.path.join(dataset_dir, "y_test.csv")).values.ravel()

mlflow.sklearn.autolog()

with mlflow.start_run(run_name="RandomForest_Autolog"):
    print("Memulai pelatihan model...")
    
    # Parameter Model
    n_estimators = 100
    random_state = 42
    
    # Melatih model
    rf_model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    rf_model.fit(X_train, y_train)
    
    # Prediksi & Evaluasi
    y_pred = rf_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Pelatihan selesai! Akurasi: {accuracy}")
