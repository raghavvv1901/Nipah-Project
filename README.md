# 🦇 Nipah Virus Spillover Risk Predictor

[![Python](https://img.shields.io/badge/Python-3.9-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Deployed-red.svg)](https://streamlit.io/)
[![GitHub Actions](https://img.shields.io/badge/GitHub%20Actions-Automated-brightgreen.svg)](https://github.com/features/actions)

**Live Interactive Dashboard:** [https://nipah-project-nhxdfim72uha6hijgvyjbs.streamlit.app](https://nipah-project-nhxdfim72uha6hijgvyjbs.streamlit.app)

## 🌍 Overview

This project is an autonomous, cloud-hosted machine learning pipeline designed to predict the real-time risk of **Nipah Virus (NiV)** spillover events from *Pteropus* bats to human populations.

Rather than relying on static, outdated epidemiological datasets, this system utilizes a continuous data pipeline. It pulls live bat telemetry, cross-references it with real-time remote sensing environmental data, and feeds it into a machine learning model to map localized spillover probabilities.

## ✨ Key Features

* **Live Ecological Monitoring:** Automatically fetches real-time bat observation coordinates using the **iNaturalist API**.
* **Remote Sensing Integration:** Connects to **Google Earth Engine** to extract localized climatic and environmental variables (Maximum Temperature, Precipitation, Tree Cover, and Population Density).
* **Automated CI/CD Pipeline:** Uses **GitHub Actions** to run a headless data-fetching server every midnight (UTC), completely bypassing manual data entry.
* **Machine Learning Prediction (v6):** Employs a **Random Forest Regressor** (`nipah_ai_v6.pkl`) to evaluate complex, non-linear environmental stress factors.
* **Explainable AI (XAI):** Features dynamic feature importance tracking, providing a transparent breakdown of exactly *why* a region is flagged as high-risk by displaying the weighted impact of environmental inputs.
* **Persistent UI State Management:** Engineered Streamlit `session_state` caching to ensure continuous, interruption-free user interactions between the Folium map and environmental sliders.

## 🧠 The Mathematical Model

To establish a biological ground-truth for the AI to train on, baseline spillover risk was modeled using an epidemiological **Logistic Function (S-Curve)**. This represents how biological stress (like habitat loss and heat) compounds exponentially after certain thresholds are crossed:

$$P(risk) = \frac{1}{1 + e^{-2.5(Score - 4.5)}}$$

**Probability Normalization:**
To prevent unbound extrapolation from the Random Forest on extreme environmental anomalies (which can result in logits exceeding 100%), the raw predictive outputs are passed through a Sigmoid activation layer at inference time. This guarantees a mathematically sound probability strictly bounded between 0% and 100%.

## ⚙️ Tech Stack & Dependencies

To ensure long-term stability, this project uses pinned library versions:
* **Core:** `Python 3.9+`, `pandas`, `numpy`
* **Machine Learning:** `scikit-learn` (Random Forest), `joblib`
* **Geospatial & UI:** `streamlit`, `folium`, `streamlit-folium`
* **APIs:** `earthengine-api`, `requests`

## 🚀 How the Autonomous Pipeline Works

1. **The Trigger:** A CRON job in `.github/workflows/update.yml` runs daily at midnight.
2. **Authentication:** The GitHub runner accesses a secure vault (GitHub Secrets) to bypass Google Earth Engine's OAuth barriers without a browser.
3. **Execution:** `dataset_builder.py` is executed, fetching live coordinates and remote sensing data.
4. **Update & Deploy:** The updated `nipah_spillover_data.csv` is committed back to the repository, which automatically triggers Streamlit to reboot the live dashboard with the newest data.
