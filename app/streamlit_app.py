"""
Simplified Bengaluru Real Estate Price Predictor using baseline Random Forest.
Uses only structured features (no NLP embeddings) for fast prediction.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium

from src.data.bengaluru_localities import (
    BENGALURU_LOCALITIES,
    BENGALURU_CENTER,
    find_nearest_locality,
    BENGALURU_REFERENCE_POINTS,
)
from src.utils.currency import format_inr
from src.features.engineer import engineer_features
from src.features.preprocessor import StructuredPreprocessor, inverse_transform_target
from src.utils.io import load_pickle
from src.utils.logger import get_logger

logger = get_logger(__name__)

st.set_page_config(
    page_title="Bengaluru Real Estate Intelligence",
    page_icon="🏠",
    layout="wide",
)


@st.cache_resource
def load_models():
    """Load trained models and preprocessors."""
    try:
        kmeans = load_pickle(Path("artifacts/preprocessors/location_kmeans.pkl"))
        preprocessor = StructuredPreprocessor.load()
        model = load_pickle(Path("artifacts/models/baseline_random_forest.pkl"))
        return kmeans, preprocessor, model
    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        return None, None, None


def create_bengaluru_map(center_lat, center_lon, selected_locality=None):
    """Create an interactive Folium map of Bengaluru with locality markers."""
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=12,
        tiles="OpenStreetMap",
    )

    if selected_locality:
        folium.Marker(
            [center_lat, center_lon],
            popup=f"<b>{selected_locality}</b>",
            tooltip=selected_locality,
            icon=folium.Icon(color="red", icon="home", prefix="fa"),
        ).add_to(m)

    it_hubs = {
        "MG Road CBD": BENGALURU_REFERENCE_POINTS["cbd"],
        "ITPL Whitefield": BENGALURU_REFERENCE_POINTS["itpl_whitefield"],
        "Electronic City": BENGALURU_REFERENCE_POINTS["electronic_city"],
        "Manyata Tech Park": BENGALURU_REFERENCE_POINTS["manyata_tech_park"],
        "Bellandur ORR": BENGALURU_REFERENCE_POINTS["bellandur_ecospace"],
        "BLR Airport": BENGALURU_REFERENCE_POINTS["airport"],
    }

    for name, (lat, lon) in it_hubs.items():
        folium.Marker(
            [lat, lon],
            popup=f"<b>{name}</b>",
            tooltip=name,
            icon=folium.Icon(color="blue", icon="building", prefix="fa"),
        ).add_to(m)

    return m


def main():
    st.title("🏠 Bengaluru Real Estate Intelligence System")
    st.markdown("""
    **ML-Powered Price Prediction**

    Predicts Bengaluru property prices using Random Forest trained on 15,000 real estate listings.
    """)

    kmeans, preprocessor, model = load_models()

    if model is None:
        st.error("⚠️ Models not loaded. Please run: `python -m src.training.train_baseline`")
        st.stop()

    st.sidebar.header("Property Location")

    location_mode = st.sidebar.radio(
        "Choose location method:",
        ["🔍 Select from Locality List", "🗺️ Click on Map"],
        index=0,
    )

    selected_locality = None
    latitude, longitude = BENGALURU_CENTER

    if location_mode == "🔍 Select from Locality List":
        locality_list = sorted(BENGALURU_LOCALITIES.keys())
        selected_locality = st.sidebar.selectbox(
            "Select Locality",
            options=locality_list,
            index=locality_list.index("Koramangala") if "Koramangala" in locality_list else 0,
        )
        latitude, longitude = BENGALURU_LOCALITIES[selected_locality]
        st.sidebar.success(f"📍 **{selected_locality}**")
        st.sidebar.caption(f"Coordinates: {latitude:.4f}, {longitude:.4f}")
    else:
        st.sidebar.info("👇 Click anywhere on the map to select location")
        map_obj = create_bengaluru_map(BENGALURU_CENTER[0], BENGALURU_CENTER[1])
        map_data = st_folium(map_obj, width=350, height=400, key="location_map")

        if map_data and map_data.get("last_clicked"):
            clicked_lat = map_data["last_clicked"]["lat"]
            clicked_lon = map_data["last_clicked"]["lng"]
            latitude, longitude = clicked_lat, clicked_lon
            nearest_loc, distance = find_nearest_locality(latitude, longitude)
            selected_locality = nearest_loc
            st.sidebar.success(f"📍 **Nearest: {nearest_loc}**")
            st.sidebar.caption(f"({distance:.2f} km from {nearest_loc})")
            st.sidebar.caption(f"Coordinates: {latitude:.4f}, {longitude:.4f}")
        else:
            selected_locality = "Koramangala"
            latitude, longitude = BENGALURU_LOCALITIES[selected_locality]
            st.sidebar.info(f"Default: {selected_locality}")

    st.sidebar.header("🏡 Property Details")
    size = st.sidebar.slider("BHK (Bedrooms)", 1, 5, 3)
    total_sqft = st.sidebar.number_input(
        "Total Area (Square Feet)",
        min_value=400.0,
        max_value=5000.0,
        value=1500.0,
        step=50.0,
    )
    bath = st.sidebar.slider("Bathrooms", 1, 5, 2)
    balcony = st.sidebar.slider("Balconies", 0, 3, 1)

    if st.sidebar.button("🔮 Predict Price", type="primary", use_container_width=True):
        property_df = pd.DataFrame([{
            "location": selected_locality,
            "latitude": latitude,
            "longitude": longitude,
            "size": size,
            "total_sqft": total_sqft,
            "bath": bath,
            "balcony": balcony,
        }])

        with st.spinner("🔄 Calculating property value..."):
            try:
                # Engineer features
                property_eng, _ = engineer_features(property_df, kmeans_model=kmeans, is_train=False)

                # Preprocess
                X_struct = preprocessor.transform(property_eng)

                # Predict
                y_pred_log = model.predict(X_struct)
                predicted_price = float(inverse_transform_target(y_pred_log)[0])

                # Confidence interval (±10% for demo)
                lower_price = predicted_price * 0.90
                upper_price = predicted_price * 1.10

            except Exception as e:
                st.error(f"Prediction failed: {e}")
                logger.error(f"Prediction error: {e}", exc_info=True)
                st.stop()

        st.success("✅ **Prediction Complete!**")
        st.markdown("---")
        st.markdown("### 💰 Predicted Property Price")

        price_col1, price_col2, price_col3 = st.columns([2, 2, 1])

        with price_col1:
            st.markdown(f"## {format_inr(predicted_price)}")
            st.caption("Estimated Market Value")

        with price_col2:
            st.metric(
                "Price Range (±10%)",
                f"{format_inr(lower_price)} - {format_inr(upper_price)}",
            )

        with price_col3:
            price_per_sqft = predicted_price / total_sqft
            st.metric("Per Sqft", f"₹{price_per_sqft:,.0f}")

        st.info("ℹ️ **Model:** Random Forest trained on 10,837 Bengaluru properties (R² = 0.9995)")

        st.markdown("---")
        st.subheader("📋 Property Summary")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("📍 Location", selected_locality)
        with col2:
            st.metric("🏠 Configuration", f"{size} BHK, {bath} Bath")
        with col3:
            st.metric("📐 Area", f"{total_sqft:,.0f} sqft")
        with col4:
            st.metric("🪟 Balconies", balcony)

        st.markdown("---")
        st.subheader("📍 Property Location on Map")
        main_map = create_bengaluru_map(latitude, longitude, selected_locality)
        st_folium(main_map, width=None, height=450, key="main_map")

    else:
        st.info("👈 Configure property details in the sidebar and click **Predict Price**")
        st.subheader("🗺️ Bengaluru IT Hubs & Major Localities")
        overview_map = create_bengaluru_map(BENGALURU_CENTER[0], BENGALURU_CENTER[1])
        st_folium(overview_map, width=900, height=500, key="overview_map")

    st.markdown("---")
    st.markdown(f"""
    **About**: Bengaluru Real Estate Intelligence System predicts property prices using:
    - **Structured Features**: BHK, sqft, bathrooms, balconies, location
    - **Distance Engineering**: Proximity to CBD (MG Road), IT hubs (ITPL, Electronic City), BLR Airport
    - **Location Clustering**: KMeans clustering on geographic coordinates
    - **Model**: Random Forest (200 trees) trained on 10,837 properties

    **Coverage**: {len(BENGALURU_LOCALITIES)} localities across North, South, East, West Bengaluru

    Built with: Python, scikit-learn, Streamlit, Folium
    """)


if __name__ == "__main__":
    main()
