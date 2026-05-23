import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os

def train_model():
    working_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(working_dir, 'Iris.csv')
    
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    

    X = df.drop(columns=['Id', 'Species'])
    y = df['Species']
    

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    
    print("Training Random Forest model on Iris dataset...")
    pipeline.fit(X_train, y_train)
    

    predictions = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"\nModel trained successfully!")
    print(f"Test Accuracy: {accuracy * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, predictions))
    

    model_save_path = os.path.join(working_dir, 'iris_rf_model.joblib')
    joblib.dump(pipeline, model_save_path)
    print(f"\nSaved the model to: {model_save_path}")

if __name__ == '__main__':
    train_model()
