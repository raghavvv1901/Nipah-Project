import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import joblib

print("🧪 Generating Final, Perfect Risk Data...")
np.random.seed(42)

# Generate 20,000 diverse scenarios
n_samples = 20000
df = pd.DataFrame({
    'Max_Temp_C': np.random.uniform(0, 50, n_samples),
    'Precipitation_mm': np.random.uniform(0, 500, n_samples),
    'Tree_Cover_Pct': np.random.uniform(0, 100, n_samples),
    'Population_Density': np.random.uniform(0, 5000, n_samples)
})

# THE PERFECT FORMULA FOR SMOOTH PERCENTAGES
def calculate_exact_risk(row):
    # 1. Weather (Bats need warmth and rain)
    if 20 <= row['Max_Temp_C'] <= 40 and row['Precipitation_mm'] >= 50:
        weather = 1.0
    else:
        weather = 0.1 # Almost zero risk if bats can't survive
        
    # 2. Habitat Disruption (Fewer trees = higher disruption)
    disruption = (100 - row['Tree_Cover_Pct']) / 100.0
    
    # 3. Human Contact (More people = higher contact, capping at 3000)
    contact = min(row['Population_Density'] / 3000.0, 1.0)
    
    # 4. Final Risk Score (Blends weather, disruption, and humans perfectly)
    risk = weather * ((disruption * 0.5) + (contact * 0.5))
    return risk

df['Risk_Percentage'] = df.apply(calculate_exact_risk, axis=1)

print("🌲 Training the Final Regressor AI...")
X = df[['Max_Temp_C', 'Precipitation_mm', 'Tree_Cover_Pct', 'Population_Density']]
y = df['Risk_Percentage']

# Regressors output smooth, exact numbers instead of rigid categories!
model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
model.fit(X, y)

joblib.dump(model, 'nipah_spillover_ai_final.pkl')
print("✅ SUCCESS: 'nipah_spillover_ai_final.pkl' created!")