# Soil Pollution & Disease AI System

A machine learning project for predicting **soil pollutant concentrations** and **disease types** associated with heavy metal contamination.
The system includes full ML pipelines, scientific visualizations, and an interactive Streamlit application.

---

## Overview

This project provides:

* A scientifically grounded synthetic dataset for soil contamination
* ML models for:

  * Pollutant concentration prediction (regression)
  * Disease type prediction (classification)
* Automated data visualizations for soil chemistry, exposure, and health patterns
* A Streamlit interface for interactive use
* Complete documentation and scientific references


---

## Project Structure

```
project-root/
│
├── app.py                          # Streamlit application
├── concentration_ai.py             # Regression model training
├── disease_ai.py                   # Classification model training
├── visualization.py                # Scientific dataset visualizations
├── utils.py                        # Utility functions
│
├── data/
│   └── soil_contamination_scientific.csv
│
├── models/                         # Auto-generated trained model files
│   ├── concentration_model.pkl
│   └── disease_type_model.pkl
│
└── visualizations/                 # Auto-generated dataset plots
```

---

## Dataset Summary

The dataset contains:

* 2,000 scientifically realistic contamination cases
* 28 variables describing:

  * Soil chemistry
  * Environmental factors
  * Agricultural context
  * Industrial contamination
  * Bioavailability
  * Health outcomes (disease type, severity, symptoms)
  * Age and gender exposure patterns

The dataset is synthetic but constructed using real-world toxicological models, mobility rules, regulatory limits, and exposure biology.

---

## Machine Learning Models

### 1. Pollutant Concentration Model

Source: 

Predicts **Total_Concentration_mg_kg** using 17 features including:

* Soil pH, texture, SOM, CEC
* Climate variables (temperature, humidity, rainfall)
* Region and country
* Industry proximity
* Crop type and farming practice
* Water source
* Distance from contamination source

Algorithms trained and compared:

* Random Forest
* Gradient Boosting
* XGBoost
* LightGBM
* Ridge
* Lasso

Produces:

* Model comparison plots
* Residual analysis
* Feature importance ranking
* Saved best-performing model

---

### 2. Disease Type Prediction Model

Source: 

Predicts **Disease_Type** (11 classes) based on:

* Total and bioavailable concentration
* Soil properties
* Climate/environment
* Health symptoms
* Age and gender vulnerability
* Exposure duration
* Industry and geography

Algorithms trained:

* Random Forest
* Gradient Boosting
* XGBoost
* LightGBM

Outputs:

* Accuracy, F1-score, precision, recall
* Per-class metrics
* Confusion matrices
* Model comparison visualization
* Saved best-performing classifier

---

## Scientific Visualizations

Source: 

Generated plots:

* pH vs Bioavailability Analysis
* Soil Texture Effects
* Distance Decay Behavior
* Pollutant Distribution
* Age Vulnerability
* Industry Contamination Patterns

All plots are automatically saved to `/visualizations`.

---

## Streamlit Application

Source: 

Provides:

* Pollutant concentration predictor
* Disease type predictor
* Dataset visualization viewer
* Model performance dashboard
* Automatic model loading and validation

Run the app:

```bash
streamlit run app.py
```

---

## How to Run the Project

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the ML Models

```bash
python concentration_ai.py
python disease_ai.py
```

Models will be saved to the `/models` directory automatically.

### 3. Generate Scientific Visualizations

```bash
python visualization.py
```

Images will be saved to `/visualizations`.

### 4. Launch the Streamlit Interface

```bash
streamlit run app.py
```

---

## Key Scientific Principles Implemented

The dataset and models integrate:

* pH–metal mobility rules
* Bioavailability models for cationic and anionic metals
* Soil texture and organic matter effects
* Distance decay formulas
* Exposure duration multipliers
* Age-specific vulnerability factors
* Crop bioaccumulation coefficients
* Industry-specific concentration profiles

---

## Limitations

* Dataset is synthetic and should not be used for regulatory or clinical decision-making
* Real-world contamination is more complex than simulated models
* Predictions represent educational research outputs only

---

## License

This project is intended for educational and research purposes.
Always consult scientific literature and environmental health experts for real-world applications.

---

