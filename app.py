import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Will it Rain Tomorrow?", page_icon="🌧️", layout="centered")


# Custom styling: soft, near-white blue background + card-like containers
st.markdown(
    """
    <style>
    .stApp {
        background-color: #f4f9fc;
    }
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background-color: #ffffff;
        border-radius: 12px;
        padding: 0.5rem 0.5rem 0.5rem 0.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# Load model and fitted preprocessors (once, cached)
@st.cache_resource
def load_model_bundle():
    return joblib.load('models/rf_model.joblib')

model_data = load_model_bundle()

model = model_data['model']
imputer = model_data['imputer']
scaler = model_data['scaler']
encoder = model_data['encoder']
input_cols = model_data['input_cols']
numeric_cols = list(model_data['numeric_cols'])
target_col = model_data['target_col']
classes = list(model_data['classes'])
categorical_cols = [
    c for c in model_data['categorical_cols']
    if c not in ('Date', target_col)
]


# Preprocess
def preprocess_input(raw_input):
    df = pd.DataFrame([raw_input])

    # Impute missing numeric values
    df[numeric_cols] = imputer.transform(df[numeric_cols])

    # Scale numeric features
    df[numeric_cols] = scaler.transform(df[numeric_cols])

    # One-hot encode categorical features
    encoded = encoder.transform(df[categorical_cols])
    encoded_cols = encoder.get_feature_names_out(categorical_cols)
    encoded_df = pd.DataFrame(encoded, columns=encoded_cols, index=df.index)

    df = pd.concat([df.drop(columns=categorical_cols), encoded_df], axis=1)

    df = df.reindex(columns=input_cols, fill_value=0)

    return df


def predict(input_df: pd.DataFrame):
    prediction = model.predict(input_df)[0]
    probabilities = model.predict_proba(input_df)[0]
    prob_of_prediction = probabilities[classes.index(prediction)]
    return prediction, prob_of_prediction


wind_directions = {
    "N": "North", "NNE": "North-Northeast", "NE": "Northeast",
    "ENE": "East-Northeast", "E": "East", "ESE": "East-Southeast",
    "SE": "Southeast", "SSE": "South-Southeast", "S": "South",
    "SSW": "South-Southwest", "SW": "Southwest", "WSW": "West-Southwest",
    "W": "West", "WNW": "West-Northwest", "NW": "Northwest",
    "NNW": "North-Northwest"
}


# Interface
st.title('🌦️ Will it Rain Tomorrow in Australia?')
st.markdown(
    "Enter today's weather data, and a Random Forest model will predict "
    "whether it will rain tomorrow."
)

# Location & current conditions
with st.container(border=True):
    st.subheader("📍 Location & Current Conditions")
    col1, col2 = st.columns(2)
    with col1:
        locations = sorted(
            c.replace('Location_', '') for c in input_cols if c.startswith('Location_')
        )
        location = st.selectbox("Location", locations)
    with col2:
        rain_today = st.radio("Is it raining today?", ["Yes", "No"], index=None, horizontal=True)

# Wind direction
with st.container(border=True):
    st.subheader("🧭 Wind Direction")
    col1, col2, col3 = st.columns(3)
    with col1:
        windgustdir = st.selectbox(
            "Gust direction",
            options=list(wind_directions.keys()),
            format_func=lambda x: f"{x} — {wind_directions[x]}"
        )
    with col2:
        wind9amdir = st.selectbox(
            "At 9 AM",
            options=list(wind_directions.keys()),
            format_func=lambda x: f"{x} — {wind_directions[x]}"
        )
    with col3:
        wind3pmdir = st.selectbox(
            "At 3 PM",
            options=list(wind_directions.keys()),
            format_func=lambda x: f"{x} — {wind_directions[x]}"
        )

# Wind speed
with st.container(border=True):
    st.subheader("💨 Wind Speed (km/h)")
    col1, col2, col3 = st.columns(3)
    with col1:
        wind_gust_speed = st.slider("Gust speed", 0, 150, 35)
    with col2:
        wind_speed_9am = st.slider("At 9 AM", 0, 100, 15)
    with col3:
        wind_speed_3pm = st.slider("At 3 PM", 0, 100, 20)

# Temperature
with st.container(border=True):
    st.subheader("🌡️ Temperature (°C)")
    col1, col2 = st.columns(2)
    with col1:
        min_temp = st.number_input("Min Temperature", value=10.0, step=0.1)
        temp_9am = st.number_input("Temperature at 9 AM", value=15.0, step=0.1)
    with col2:
        max_temp = st.number_input("Max Temperature", value=20.0, step=0.1)
        temp_3pm = st.number_input("Temperature at 3 PM", value=22.0, step=0.1)

# Precipitation & sunshine
with st.container(border=True):
    st.subheader("🌧️ Precipitation & Sunshine")
    col1, col2, col3 = st.columns(3)
    with col1:
        rainfall = st.number_input("Rainfall (mm)", min_value=0.0, value=0.0, step=0.1)
    with col2:
        evaporation = st.number_input("Evaporation (mm)", min_value=0.0, value=5.0, step=0.1)
    with col3:
        sunshine = st.slider("Sunshine (hours)", 0.0, 15.0, 8.0, 0.1)

# Humidity & pressure
with st.container(border=True):
    st.subheader("💧 Humidity & Pressure")
    col1, col2 = st.columns(2)
    with col1:
        humidity_9am = st.slider("Humidity at 9 AM (%)", 0, 100, 70)
        pressure_9am = st.number_input("Pressure at 9 AM (hPa)", value=1015.0, step=0.1)
    with col2:
        humidity_3pm = st.slider("Humidity at 3 PM (%)", 0, 100, 50)
        pressure_3pm = st.number_input("Pressure at 3 PM (hPa)", value=1013.0, step=0.1)

# Cloud cover
with st.container(border=True):
    st.subheader("☁️ Cloud Cover (oktas)")
    col1, col2 = st.columns(2)
    with col1:
        cloud_9am = st.slider("At 9 AM", 0, 8, 4)
    with col2:
        cloud_3pm = st.slider("At 3 PM", 0, 8, 4)



# Prediction
st.divider()

if st.button("🔮 Predict", type="primary", use_container_width=True):
    if rain_today is None:
        st.warning("Please select whether it is raining today.")
    else:
        raw_input = {
            'Location': location,
            'MinTemp': min_temp,
            'MaxTemp': max_temp,
            'Rainfall': rainfall,
            'Evaporation': evaporation,
            'Sunshine': sunshine,
            'WindGustDir': windgustdir,
            'WindGustSpeed': wind_gust_speed,
            'WindDir9am': wind9amdir,
            'WindDir3pm': wind3pmdir,
            'WindSpeed9am': wind_speed_9am,
            'WindSpeed3pm': wind_speed_3pm,
            'Humidity9am': humidity_9am,
            'Humidity3pm': humidity_3pm,
            'Pressure9am': pressure_9am,
            'Pressure3pm': pressure_3pm,
            'Cloud9am': cloud_9am,
            'Cloud3pm': cloud_3pm,
            'Temp9am': temp_9am,
            'Temp3pm': temp_3pm,
            'RainToday': rain_today,
        }

        input_df = preprocess_input(raw_input)
        prediction, probability = predict(input_df)

        st.subheader("Prediction Result")

        if prediction == "Yes":
            st.error(f"🌧️ It will likely rain tomorrow (probability: {probability:.1%})")
        else:
            st.success(f"☀️ It will likely NOT rain tomorrow (probability: {probability:.1%})")