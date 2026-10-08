"""
QSkill AI/ML Internship - Task 3: House Price Prediction
Streamlit UI — enter house/location details and get an instant predicted
median house value.

Run with:
    streamlit run app.py
(Run train_model.py first if pipeline.pkl / defaults.pkl don't exist yet.)
"""

import os
import joblib
import pandas as pd
import streamlit as st


@st.cache_resource
def load_artifacts():
    if not (os.path.exists("pipeline.pkl") and os.path.exists("defaults.pkl")):
        return None, None, None
    pipeline = joblib.load("pipeline.pkl")
    defaults = joblib.load("defaults.pkl")
    model_name = "Model"
    if os.path.exists("model_name.txt"):
        with open("model_name.txt") as f:
            model_name = f.read().strip()
    return pipeline, defaults, model_name


st.set_page_config(page_title="House Price Predictor", page_icon="🏠", layout="centered")

st.title("🏠 House Price Predictor")
st.caption("QSkill AI/ML Internship — Task 3")

pipeline, defaults, model_name = load_artifacts()

if pipeline is None:
    st.error(
        "Model files not found. Run `python3 train_model.py` first "
        "(with housing.csv in the same folder) to generate pipeline.pkl."
    )
    st.stop()

st.markdown(f"**Model in use:** {model_name} · trained on California housing block-group data")

st.subheader("Location")
col1, col2 = st.columns(2)
with col1:
    latitude = st.number_input("Latitude", value=float(defaults["latitude"]), format="%.2f")
with col2:
    longitude = st.number_input("Longitude", value=float(defaults["longitude"]), format="%.2f")

ocean_proximity = st.selectbox("Ocean proximity", defaults["ocean_proximity_options"])

st.subheader("House / block-group details")
col3, col4 = st.columns(2)
with col3:
    housing_median_age = st.slider("Median house age (years)", 1, 52, int(defaults["housing_median_age"]))
    total_rooms = st.number_input("Total rooms (block group)", min_value=1, value=int(defaults["total_rooms"]))
    total_bedrooms = st.number_input("Total bedrooms (block group)", min_value=1, value=int(defaults["total_bedrooms"]))
with col4:
    population = st.number_input("Population (block group)", min_value=1, value=int(defaults["population"]))
    households = st.number_input("Households (block group)", min_value=1, value=int(defaults["households"]))
    median_income = st.slider("Median income (10k$ units)", 0.5, 15.0, float(defaults["median_income"]))

if st.button("Predict price", type="primary", use_container_width=True):
    input_df = pd.DataFrame([{
        "longitude": longitude,
        "latitude": latitude,
        "housing_median_age": housing_median_age,
        "total_rooms": total_rooms,
        "total_bedrooms": total_bedrooms,
        "population": population,
        "households": households,
        "median_income": median_income,
        "ocean_proximity": ocean_proximity,
    }])

    prediction = pipeline.predict(input_df)[0]
    st.success(f"### Predicted median house value: ${prediction:,.0f}")
    st.caption(
        "Note: predictions are for a census block group (a neighborhood-level average), "
        "not a single house — this matches how the training data (1990 California census) is structured."
    )

st.divider()
with st.expander("About this dataset"):
    st.write(
        "Trained on the California Housing Prices dataset (1990 census, Pace & Barry 1997): "
        "20,640 block groups with features on location, house age, rooms/bedrooms, "
        "population, households, income, and proximity to the ocean."
    )
