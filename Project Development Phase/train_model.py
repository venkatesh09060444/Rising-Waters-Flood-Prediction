import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import xgboost as xgb
import os

def train_and_evaluate_models(data_file='flood_data.csv', model_file='best_model.pkl'):
    if not os.path.exists(data_file):
        print(f"Error: {data_file} not found. Please run generate_data.py first.")
        return

    print("Loading data...")
    df = pd.read_csv(data_file)
    
    X = df[['annual_rainfall', 'cloud_visibility', 'seasonal_rainfall']]
    y = df['flood']
    
    print("Splitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    models = {
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'KNN': KNeighborsClassifier(n_neighbors=5),
        'XGBoost': xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    }
    
    best_accuracy = 0
    best_model = None
    best_model_name = ""
    
    print("\\nTraining and evaluating models...")
    for name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        print(f"{name} Accuracy: {accuracy * 100:.2f}%")
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model
            best_model_name = name
            
    print(f"\\nBest model is {best_model_name} with {best_accuracy * 100:.2f}% accuracy.")
    
    # Save the best model
    joblib.dump(best_model, model_file)
    print(f"Model saved to {model_file}")

if __name__ == '__main__':
    train_and_evaluate_models()
