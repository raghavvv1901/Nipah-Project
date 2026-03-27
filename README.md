# 🦇 Nipah Virus Spillover Risk Predictor

[![Python](https://img.shields.io/badge/Python-3.9-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Deployed-red.svg)](https://streamlit.io/)
[![GitHub Actions](https://img.shields.io/badge/GitHub%20Actions-Automated-brightgreen.svg)](https://github.com/features/actions)

**Live Interactive Dashboard:** [Insert your Streamlit App URL here]

## 🌍 Overview
This project is an autonomous, cloud-hosted machine learning pipeline designed to predict the real-time risk of **Nipah Virus (NiV)** spillover events from *Pteropus* bats to human populations. 

Rather than relying on static, outdated epidemiological datasets, this system utilizes a continuous data pipeline. It pulls live bat telemetry, cross-references it with real-time remote sensing environmental data, and feeds it into a machine learning model to map localized spillover probabilities.

## ✨ Key Features
* **Live Ecological Monitoring:** Automatically fetches real-time bat observation coordinates using the **iNaturalist API**.
* **Remote Sensing Integration:** Connects to **Google Earth Engine** to extract localized climatic and environmental variables (Maximum Temperature, Precipitation, Tree Cover, and Population Density).
* **Automated CI/CD Pipeline:** Uses **GitHub Actions** to run a headless data-fetching server every midnight (UTC), completely bypassing manual data entry.
* **Machine Learning Prediction:** Employs a **Random Forest Regressor** to evaluate complex, non-linear environmental stress factors and output a spillover probability percentage.
* **Interactive UI:** A real-time epidemiological dashboard built with **Streamlit** and **Folium** mapping.

## 🧠 The Mathematical Model
To establish a biological ground-truth for the AI to train on, spillover risk was modeled using an epidemiological **Logistic Function (S-Curve)**. This represents how biological stress (like habitat loss and heat) compounds exponentially after certain thresholds are crossed.

$$P(risk) = \frac{1}{1 + e^{-2.5(Score - 4.5)}}$$

This formula forces the baseline data into strict "Safe Zones" (0-5%) or "Critical Zones" (>90%), training the Random Forest to recognize the exact tipping point of environmental collapse.

## ⚙️ Tech Stack
* **Language:** Python
* **Cloud & Automation:** GitHub Actions, Streamlit Community Cloud
* **APIs & Data:** Google Earth Engine (`earthengine-api`), iNaturalist API
* **Machine Learning:** `scikit-learn` (Random Forest Regressor), `joblib`, `pandas`, `numpy`
* **Geospatial & UI:** `folium`, `streamlit-folium`

## 🚀 How the Autonomous Pipeline Works
1. **The Trigger:** A CRON job in `.github/workflows/update.yml` runs daily at midnight.
2. **Authentication:** The GitHub runner accesses a secure vault (GitHub Secrets) to bypass Google Earth Engine's OAuth barriers without a browser.
3. **Execution:** `dataset_builder.py` is executed, fetching live coordinates and remote sensing data.
4. **Update & Deploy:** The updated `nipah_spillover_data.csv` is committed back to the repository, which automatically triggers Streamlit to reboot the live dashboard with the newest data.
