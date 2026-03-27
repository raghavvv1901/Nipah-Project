import streamlit as st
import pandas as pd
import joblib
import folium
from streamlit_folium import st_folium

st.set_page_config(page_title="Nipah Risk Command Center", page_icon="🦇", layout="wide")

if 'temp' not in st.session_state:
    st.session_state['temp'] = 30.0
if 'precip' not in st.session_state:
    st.session_state['precip'] = 150.0
if 'tree' not in st.session_state:
    st.session_state['tree'] = 80.0
if 'pop' not in st.session_state:
    st.session_state['pop'] = 50.0

@st.cache_resource 
def load_model():
    # LOADING THE NEW V6 FILE!
    return joblib.load("nipah_ai_v6.pkl")

@st.cache_data
def load_bat_data():
    return pd.read_csv("nipah_spillover_data.csv")

model = load_model()
try:
    df_bats = load_bat_data()
except FileNotFoundError:
    df_bats = pd.DataFrame() 

st.title("🚨 Nipah Virus Global Command Center")
st.write("Integrating real-time iNaturalist telemetry, Google Earth Engine environmental data, and Machine Learning.")
st.markdown("---")

col_map, col_ai = st.columns([2, 1])

with col_map:
    st.subheader("🗺️ Live Bat Distribution")
    m = folium.Map(location=[20.0, 78.0], zoom_start=3, tiles="CartoDB positron")
    
    if not df_bats.empty:
        for index, row in df_bats.iterrows():
            popup_info = f"""
            <b>Species:</b> {row.get('Species', 'Pteropus')}<br>
            <b>Max Temp:</b> {row.get('Max_Temp_C', 0):.1f}°C<br>
            <b>Precip:</b> {row.get('Precipitation_mm', 0):.1f}mm<br>
            <b>Tree Cover:</b> {row.get('Tree_Cover_Pct', 0):.1f}%
            """
            folium.CircleMarker(
                location=[row['Latitude'], row['Longitude']],
                radius=5,
                color="red",
                fill=True,
                fill_opacity=0.7,
                tooltip=popup_info
            ).add_to(m)
            
    map_data = st_folium(m, width=800, height=500)
    
    if map_data and map_data.get("last_object_clicked"):
        lat = map_data["last_object_clicked"]["lat"]
        lng = map_data["last_object_clicked"]["lng"]
        match = df_bats[(abs(df_bats['Latitude'] - lat) < 0.001) & (abs(df_bats['Longitude'] - lng) < 0.001)]
        
        if not match.empty:
            bat = match.iloc[0]
            st.session_state['temp'] = float(bat['Max_Temp_C'])
            st.session_state['precip'] = float(bat['Precipitation_mm'])
            tree = float(bat['Tree_Cover_Pct'])
            st.session_state['tree'] = tree
            st.session_state['pop'] = float(3000 - (tree * 30))

with col_ai:
    st.subheader("🧠 Spillover Simulator")
    
    temp = st.slider("Max Temperature (°C)", -10.0, 50.0, key="temp", step=0.5)
    precip = st.slider("Precipitation (mm)", 0.0, 500.0, key="precip", step=5.0)
    tree_cover = st.slider("Tree Cover (%)", 0.0, 100.0, key="tree", step=1.0)
    pop_density = st.slider("Population Density (people/km²)", 0.0, 5000.0, key="pop", step=10.0)

    user_data = pd.DataFrame({
        'Max_Temp_C': [st.session_state['temp']],
        'Precipitation_mm': [st.session_state['precip']],
        'Tree_Cover_Pct': [st.session_state['tree']],
        'Population_Density': [st.session_state['pop']]
    })

    st.markdown("---")
    if st.button("Run AI Risk Assessment", type="primary", use_container_width=True):
        risk_probability = model.predict(user_data)[0] * 100
        
        st.markdown("### 📊 Assessment Results")
        if risk_probability > 75:
            st.error(f"**CRITICAL RISK: {risk_probability:.1f}% Probability**")
        elif risk_probability > 40:
            st.warning(f"**MODERATE RISK: {risk_probability:.1f}% Probability**")
        else:
            st.success(f"**SAFE ZONE: {risk_probability:.1f}% Probability**")