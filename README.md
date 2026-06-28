# Will It Rain Tomorrow? — Australia Rain Prediction App

A Streamlit web app that predicts whether it will rain tomorrow at a given location in Australia, based on today's weather conditions. Powered by a Random Forest classifier trained on the [Australian weather dataset](https://www.kaggle.com/datasets/jsphyg/weather-dataset-rattle-package).

🔗 **Live demo:** [australianrain.streamlit.app](https://australianrain.streamlit.app/)

## Overview

The app takes today's weather observations (temperature, humidity, wind, pressure, cloud cover, etc.) as input through an interactive UI, runs them through the same preprocessing pipeline used during training, and returns a Random Forest prediction along with its probability.

## Features

- Interactive input via dropdowns, sliders, and number fields — no coding required
- Full preprocessing pipeline applied to user input (imputation → scaling → one-hot encoding)
- Random Forest prediction with confidence/probability score
- Clean, card-based UI organized into logical sections (location, wind, temperature, precipitation, humidity, pressure, cloud cover)

## Tech Stack

- **Frontend / App:** Streamlit
- **ML:** scikit-learn (RandomForestClassifier, SimpleImputer, MinMaxScaler, OneHotEncoder)
- **Data processing:** pandas, NumPy
- **Model persistence:** joblib

## Project Structure
.

├── app.py                  # Streamlit application

├── model_check.ipynb       # Notebook: data prep, training, hyperparameter tuning

├── requirements.txt        # Python dependencies

└── models/

└── rf_model.joblib      # Trained model + fitted preprocessors (imputer, scaler, encoder)

## How It Works

1. User enters today's weather data through the Streamlit UI
2. Input is converted into a single-row DataFrame
3. Preprocessing pipeline is applied — **identical to training**:
   - Missing numeric values imputed (mean strategy)
   - Numeric features scaled (MinMaxScaler)
   - Categorical features one-hot encoded
4. Processed row is fed into the trained Random Forest model
5. App displays the prediction (`Yes` / `No`) and the associated probability

## Model

- **Algorithm:** Random Forest Classifier
- **Tuning:** `RandomizedSearchCV` over `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`, `max_features`, `bootstrap`
- **Validation metric:** F1-score (positive class: `Yes`)
- **Train/validation/test split:** by year (pre-2015 / 2015 / post-2015)


- **F1 (test):** 0.577
- **ROC AUC (test):** 0.8609


Full training process, EDA, and evaluation (confusion matrix, ROC curve) are documented in [`rain_predict.ipynb`](./rain_predict.ipynb).

## Running Locally

```bash
git clone https://github.com/your-username/your-repo.git
cd your-repo
pip install -r requirements.txt
streamlit run app.py
```

The app will open at `http://localhost:8501`.

## Deployment

Deployed via [Streamlit Community Cloud](https://share.streamlit.io). To deploy your own copy:
1. Fork/clone this repo
2. Push to your own GitHub repository
3. Connect the repo on Streamlit Community Cloud, pointing to `app.py`

## Dataset

[Rain in Australia](https://www.kaggle.com/datasets/jsphyg/weather-dataset-rattle-package) — daily weather observations from numerous Australian weather stations, sourced from the Australian Bureau of Meteorology.
