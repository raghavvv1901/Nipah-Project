import pandas as pd
import requests
import ee
import math
from sklearn.ensemble import RandomForestRegressor
import joblib

print("🌍 Initializing Earth Engine...")
ee.Initialize(project='tusk-trust')

print("🦇 Fetching live bat locations from iNaturalist...")
url = "https://api.inaturalist.org/v1/observations"
params = {
    "taxon_name": "Pteropus",
    "quality_grade": "research",
    "has[]": "geo",
    "per_page": 100
}
headers = {"User-Agent": "NipahRiskProject/1.0"}

records = []
try:
    response = requests.get(url, params=params, headers=headers, timeout=15)
    response.raise_for_status() 
    data = response.json()
    for obs in data.get('results', []):
        location = obs.get('location') 
        species = obs.get('taxon', {}).get('name', 'Unknown')
        
        if location and "Pteropus" in species:
            lat_str, lon_str = location.split(',')
            records.append({
                'Species': species,
                'Latitude': float(lat_str),
                'Longitude': float(lon_str),
                'Bat_Presence': 1 
            })
            
except Exception as e:
    print(f"❌ API Failure: {e}")
    exit()

df = pd.DataFrame(records).drop_duplicates(subset=['Latitude', 'Longitude'])
print(f"✅ Successfully pulled {len(df)} live bat locations!")

print("🛰️ Querying Earth Engine for Climate, Population, and Forest Data...")

# 1. Weather Data (TerraClimate)
climate = ee.ImageCollection('IDAHO_EPSCOR/TERRACLIMATE').filterDate('2023-01-01', '2023-12-31').mean()

# 2. Human Population Data (WorldPop)
population = ee.ImageCollection("WorldPop/GP/100m/pop").filterDate('2020-01-01', '2020-12-31').mean()

# 3. Forest Cover Data (Hansen Global Forest Change)
forest = ee.Image("UMD/hansen/global_forest_change_2023_v1_11")

max_temps, precips, pop_densities, tree_covers = [], [], [], []
total_bats = len(df)

for index, row in df.iterrows():
    if index % 10 == 0 and index > 0:
        print(f"   ... Scanned {index} out of {total_bats} locations ...")
        
    poi = ee.Geometry.Point([row['Longitude'], row['Latitude']])
    
    try:
        # Get Climate
        clim_data = climate.reduceRegion(reducer=ee.Reducer.mean(), geometry=poi, scale=4000).getInfo()
        temp_c = clim_data.get('tmmx')
        temp_c = temp_c * 0.1 if temp_c is not None else None
        precip_mm = clim_data.get('pr')
        
        # Get Population Density
        pop_data = population.reduceRegion(reducer=ee.Reducer.mean(), geometry=poi, scale=1000).getInfo()
        pop_val = pop_data.get('population') if pop_data and 'population' in pop_data else 0.0
        
        # Get Tree Cover Percentage
        forest_data = forest.reduceRegion(reducer=ee.Reducer.mean(), geometry=poi, scale=1000).getInfo()
        tree_val = forest_data.get('treecover2000') if forest_data and 'treecover2000' in forest_data else 0.0
        
    except Exception:
        temp_c, precip_mm, pop_val, tree_val = None, None, None, None
        
    max_temps.append(temp_c)
    precips.append(precip_mm)
    pop_densities.append(pop_val)
    tree_covers.append(tree_val)

# Add our new scientific features to the dataset
df['Max_Temp_C'] = max_temps
df['Precipitation_mm'] = precips
df['Population_Density'] = pop_densities
df['Tree_Cover_Pct'] = tree_covers

# Clean out any missing data
df_final = df.dropna().copy()

print("🧮 Calculating Normalized Spillover Risk...")

def calculate_risk(temp, precip, tree, pop):
    # Normalize features roughly between 0 and 1
    norm_temp = (temp - 20) / 20 if temp > 20 else 0  
    norm_tree = (100 - tree) / 100 # Inverse: lower tree cover = higher risk
    norm_pop = pop / 5000 
    norm_precip = precip / 300 
    
    # Calculate weighted stress score (max around 4 or 5)
    stress_score = (norm_temp * 1.5) + (norm_tree * 2.0) + (norm_pop * 1.0) + (norm_precip * 0.5)
    
    # Calculate Sigmoid Probability (Threshold at 2.5)
    probability = 1 / (1 + math.exp(-2.0 * (stress_score - 2.5)))
    return round(probability * 100, 2)

# Apply the math to every row to create the target variable
df_final['Risk_Percentage'] = df_final.apply(
    lambda row: calculate_risk(row['Max_Temp_C'], row['Precipitation_mm'], row['Tree_Cover_Pct'], row['Population_Density']), 
    axis=1
)

# Saving the balanced CSV
csv_filename = "nipah_spillover_data.csv"
df_final.to_csv(csv_filename, index=False)
print(f"✅ Saved {len(df_final)} records to '{csv_filename}'")

print("🤖 Training Machine Learning Model with normalized data...")
X = df_final[['Max_Temp_C', 'Precipitation_mm', 'Tree_Cover_Pct', 'Population_Density']]
y = df_final['Risk_Percentage']

# Train Random Forest
model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
model.fit(X, y)

# Save the trained AI
joblib.dump(model, 'nipah_ai_v6.pkl')
print("🎉 SUCCESS! New CSV and AI Model (nipah_ai_v6.pkl) have been created!")