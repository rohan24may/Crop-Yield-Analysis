
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = PROJECT_ROOT / "data" / "crop_yield.csv"

st.set_page_config(
    page_title="CropYield India",
    page_icon="🌾",
    layout="wide",
)

st.title("🌾 CropYield India")
st.write(
    "Explore historical crop yields and estimate crop yield "
    "using agricultural and climatic factors."
)

# Load the dataset
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)

    for column in ["Crop", "Season", "State"]:
        df[column] = df[column].astype(str).str.strip()

    return df


try:
    df = load_data()
except FileNotFoundError:
    st.error("Dataset not found. Check that data/crop_yield.csv exists.")
    st.stop()

# Prepare options from the real dataset
crops = sorted(df["Crop"].dropna().unique())
seasons = sorted(df["Season"].dropna().unique())
states = sorted(df["State"].dropna().unique())

historical_tab, prediction_tab = st.tabs(
    ["📚 Historical Lookup", "🔮 Predict Yield"]
)

# -----------------------------------
# Historical lookup
# -----------------------------------
with historical_tab:
    st.subheader("Look up recorded crop yield")
    st.caption(
        "This section searches the dataset for actual recorded observations. "
        "It does not generate predictions."
    )

    col1, col2 = st.columns(2)

    with col1:
        historical_crop = st.selectbox(
            "Crop", crops, key="historical_crop"
        )
        historical_year = st.number_input(
            "Year",
            min_value=int(df["Crop_Year"].min()),
            max_value=int(df["Crop_Year"].max()),
            value=2020,
            step=1,
            key="historical_year",
        )

    with col2:
        historical_season = st.selectbox(
            "Season", seasons, key="historical_season"
        )
        historical_state = st.selectbox(
            "State", states, key="historical_state"
        )

    if st.button("Search Historical Data", type="primary"):
        matches = df[
            (df["Crop"] == historical_crop)
            & (df["Crop_Year"] == historical_year)
            & (df["Season"] == historical_season)
            & (df["State"] == historical_state)
        ]

        if matches.empty:
            st.warning(
                "No matching record was found for this exact combination. "
                "This does not mean the crop was not grown there; "
                "the dataset may simply not contain that observation."
            )
        else:
            st.success(f"Found {len(matches)} matching record(s).")

            display_columns = [
                "Crop", "Crop_Year", "Season", "State",
                "Area", "Production", "Annual_Rainfall", "Yield"
            ]

            st.dataframe(
                matches[display_columns].reset_index(drop=True),
                use_container_width=True,
            )

            if len(matches) == 1:
                st.metric(
                    "Recorded Yield",
                    f"{matches.iloc[0]['Yield']:,.4f}",
                )

# -----------------------------------
# Yield prediction
# -----------------------------------
with prediction_tab:
    st.subheader("Estimate crop yield")
    st.caption(
        "This section uses your trained machine-learning model. "
        "The result is an estimate, not an actual recorded yield."
    )

    with st.form("prediction_form"):
        col1, col2 = st.columns(2)

        with col1:
            crop = st.selectbox("Crop", crops, key="prediction_crop")
            year = st.number_input(
                "Crop Year",
                min_value=int(df["Crop_Year"].min()),
                max_value=2100,
                value=2020,
                step=1,
            )
            season = st.selectbox("Season", seasons, key="prediction_season")
            state = st.selectbox("State", states, key="prediction_state")

        with col2:
            rainfall = st.number_input(
                "Annual Rainfall",
                min_value=0.0,
                value=800.0,
                help="Use the same units as the training dataset.",
            )
            fertilizer = st.number_input(
                "Fertilizer",
                min_value=0.0,
                value=100000.0,
                help="Use the same units as the training dataset.",
            )
            pesticide = st.number_input(
                "Pesticide",
                min_value=0.0,
                value=500.0,
                help="Use the same units as the training dataset.",
            )

        submitted = st.form_submit_button(
            "Predict Yield", type="primary"
        )

    if submitted:
        try:
            from src.predict import predict_crop_yield

            with st.spinner("Estimating crop yield..."):
                prediction = predict_crop_yield(
                    crop=crop,
                    crop_year=int(year),
                    season=season,
                    state=state,
                    annual_rainfall=rainfall,
                    fertilizer=fertilizer,
                    pesticide=pesticide,
                )

            st.success("Prediction completed!")
            st.metric("Estimated Crop Yield", f"{prediction:,.4f}")
            st.info(
                "This is a model estimate. Accuracy may vary by crop, "
                "state, year, and input values."
            )

        except FileNotFoundError:
            st.error(
                "Trained model not found. Run `python src/train.py` "
                "from the project root first."
            )
        except Exception as error:
            st.error(f"Prediction failed: {error}")
