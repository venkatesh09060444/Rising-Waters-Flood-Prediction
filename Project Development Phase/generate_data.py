import pandas as pd
import numpy as np

def generate_synthetic_data(num_samples=1000, output_file='flood_data.csv'):
    """
    Generates a synthetic dataset for flood prediction.
    Features:
    - annual_rainfall (mm): 500 to 4000
    - cloud_visibility (scale 0-10): 0 is clear, 10 is very cloudy/low visibility
    - seasonal_rainfall (mm): 100 to 2000
    Target:
    - flood: 1 if flooded, 0 otherwise
    """
    np.random.seed(42)
    
    annual_rainfall = np.random.uniform(500, 4000, num_samples)
    cloud_visibility = np.random.uniform(0, 10, num_samples)
    seasonal_rainfall = np.random.uniform(100, 2000, num_samples)
    
    # Create a simplistic rule for flood condition to make the model learn easily
    # High annual rainfall, high seasonal rainfall, and high cloud visibility increase risk
    risk_score = (annual_rainfall / 4000) * 0.4 + (seasonal_rainfall / 2000) * 0.4 + (cloud_visibility / 10) * 0.2
    
    # Add some noise
    risk_score += np.random.normal(0, 0.1, num_samples)
    
    # Threshold for flooding
    flood = (risk_score > 0.65).astype(int)
    
    df = pd.DataFrame({
        'annual_rainfall': annual_rainfall,
        'cloud_visibility': cloud_visibility,
        'seasonal_rainfall': seasonal_rainfall,
        'flood': flood
    })
    
    df.to_csv(output_file, index=False)
    print(f"Generated synthetic dataset with {num_samples} samples and saved to {output_file}.")
    print(f"Flood cases: {df['flood'].sum()} / {num_samples}")

if __name__ == '__main__':
    generate_synthetic_data()
