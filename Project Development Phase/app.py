from flask import Flask, request, render_template, jsonify
import joblib
import pandas as pd
import os

app = Flask(__name__)

# Load the trained model
MODEL_FILE = 'best_model.pkl'
DATA_FILE = 'flood_data.csv'

# Initialize model variable
model = None

# Get dataset averages for graphs
if os.path.exists(DATA_FILE):
    try:
        df = pd.read_csv(DATA_FILE)
        avg_annual_rainfall = float(df['annual_rainfall'].mean())
        avg_cloud_visibility = float(df['cloud_visibility'].mean())
        avg_seasonal_rainfall = float(df['seasonal_rainfall'].mean())
    except Exception:
        avg_annual_rainfall, avg_cloud_visibility, avg_seasonal_rainfall = 2250.0, 5.0, 1050.0
else:
    avg_annual_rainfall, avg_cloud_visibility, avg_seasonal_rainfall = 2250.0, 5.0, 1050.0

@app.before_request
def load_model():
    global model
    if model is None:
        if os.path.exists(MODEL_FILE):
            model = joblib.load(MODEL_FILE)
            print("Model loaded successfully.")
        else:
            print(f"Warning: {MODEL_FILE} not found. Predictions will not work until the model is trained.")

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return render_template('index.html', error="Model not found. Please train the model first.")
        
    try:
        # Extract features from the form
        annual_rainfall = float(request.form['annual_rainfall'])
        cloud_visibility = float(request.form['cloud_visibility'])
        seasonal_rainfall = float(request.form['seasonal_rainfall'])
        
        # Create a DataFrame for the model
        input_data = pd.DataFrame([[annual_rainfall, cloud_visibility, seasonal_rainfall]],
                                  columns=['annual_rainfall', 'cloud_visibility', 'seasonal_rainfall'])
        
        # Make prediction
        prediction = int(model.predict(input_data)[0])
        probability = float(model.predict_proba(input_data)[0][1]) if hasattr(model, 'predict_proba') else 0.5
        
        # Determine risk level
        if prediction == 1:
            risk = "High Risk of Flooding"
            alert_class = "alert-danger"
        else:
            risk = "Low Risk of Flooding"
            alert_class = "alert-success"
            
        return render_template('result.html', 
                               result=risk, 
                               probability=probability,
                               alert_class=alert_class, 
                               a_rain=annual_rainfall, 
                               c_vis=cloud_visibility, 
                               s_rain=seasonal_rainfall,
                               avg_a_rain=avg_annual_rainfall,
                               avg_c_vis=avg_cloud_visibility,
                               avg_s_rain=avg_seasonal_rainfall)
                               
    except Exception as e:
        return render_template('index.html', error=f"An error occurred: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)

