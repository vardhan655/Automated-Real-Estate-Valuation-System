"""
Bengaluru Real Estate Intelligence System - Streamlit Web Application.

Run with: streamlit run app/streamlit_app.py
"""

import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import folium
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

from src.data.bengaluru_localities import (
    BENGALURU_CENTER,
    BENGALURU_LOCALITIES,
    BENGALURU_REFERENCE_POINTS,
    find_nearest_locality,
)
from src.features.engineer import engineer_features
from src.features.preprocessor import StructuredPreprocessor, inverse_transform_target
from src.utils.currency import format_inr
from src.utils.io import load_pickle
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Page configuration
st.set_page_config(
    page_title="Bengaluru Real Estate Intelligence",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for clean UI styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .price-card {
        background: linear-gradient(135deg, #1E40AF 0%, #3B82F6 100%);
        padding: 1.8rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .price-title {
        font-size: 1rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        opacity: 0.9;
        margin-bottom: 0.5rem;
    }
    .price-value {
        font-size: 2.8rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
    }
    .price-subtitle {
        font-size: 0.95rem;
        opacity: 0.85;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_prediction_pipeline():
    """Load pre-trained artifacts once and cache them in memory."""
    try:
        kmeans = load_pickle(Path("artifacts/preprocessors/location_kmeans.pkl"))
        preprocessor = StructuredPreprocessor.load()
        model = load_pickle(Path("artifacts/models/baseline_random_forest.pkl"))
        return kmeans, preprocessor, model
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        return None, None, None


def create_folium_map(lat: float, lon: float, selected_locality: str = None, zoom: int = 12):
    """Create Folium map with key Bengaluru landmarks and selected property pin."""
    m = folium.Map(
        location=[lat, lon],
        zoom_start=zoom,
        tiles="OpenStreetMap",
        control_scale=True,
    )

    # Add property pin
    if selected_locality:
        folium.Marker(
            [lat, lon],
            popup=f"<b>Selected Property</b><br>{selected_locality}",
            tooltip=f"📍 {selected_locality}",
            icon=folium.Icon(color="red", icon="home", prefix="fa"),
        ).add_to(m)

    # Reference IT Hubs & Landmarks
    landmarks = {
        "MG Road CBD": (BENGALURU_REFERENCE_POINTS["cbd"], "darkblue", "briefcase"),
        "ITPL Whitefield": (BENGALURU_REFERENCE_POINTS["itpl_whitefield"], "blue", "laptop"),
        "Electronic City": (BENGALURU_REFERENCE_POINTS["electronic_city"], "blue", "laptop"),
        "Manyata Tech Park": (BENGALURU_REFERENCE_POINTS["manyata_tech_park"], "blue", "laptop"),
        "Bellandur ORR": (BENGALURU_REFERENCE_POINTS["bellandur_ecospace"], "blue", "laptop"),
        "Kempegowda Airport": (BENGALURU_REFERENCE_POINTS["airport"], "purple", "plane"),
    }

    for name, ((l_lat, l_lon), color, icon) in landmarks.items():
        folium.Marker(
            [l_lat, l_lon],
            popup=f"<b>{name}</b>",
            tooltip=name,
            icon=folium.Icon(color=color, icon=icon, prefix="fa"),
        ).add_to(m)

    return m


def main():
    st.markdown('<div class="main-header">🏠 Bengaluru Real Estate Intelligence System</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Automated property valuation powered by Random Forest regression & geospatial feature engineering.</div>', unsafe_allow_html=True)

    kmeans, preprocessor, model = load_prediction_pipeline()

    if model is None:
        st.error("⚠️ Model artifacts missing. Please train models first using `python -m src.training.train_baseline`")
        return

    # Initialize Session State
    if "predicted_result" not in st.session_state:
        st.session_state.predicted_result = None

    # Sidebar inputs
    st.sidebar.header("📍 Property Location")

    location_mode = st.sidebar.radio(
        "Location Input Method:",
        ["🔍 Select Locality from List", "🗺️ Pick on Map"],
        index=0,
    )

    selected_locality = "Koramangala"
    latitude, longitude = BENGALURU_LOCALITIES["Koramangala"]

    if location_mode == "🔍 Select Locality from List":
        locality_list = sorted(BENGALURU_LOCALITIES.keys())
        default_index = locality_list.index("Koramangala") if "Koramangala" in locality_list else 0
        selected_locality = st.sidebar.selectbox(
            "Select Locality (210+ areas)",
            options=locality_list,
            index=default_index,
        )
        latitude, longitude = BENGALURU_LOCALITIES[selected_locality]
        st.sidebar.success(f"📍 **{selected_locality}**")
        st.sidebar.caption(f"Coordinates: `{latitude:.4f}, {longitude:.4f}`")

    else:
        st.sidebar.info("Click anywhere on the map to choose coordinates:")
        picker_map = create_folium_map(BENGALURU_CENTER[0], BENGALURU_CENTER[1], zoom=11)
        map_out = st_folium(
            picker_map,
            width=300,
            height=300,
            key="sidebar_map_picker",
            returned_objects=["last_clicked"],
        )

        if map_out and map_out.get("last_clicked"):
            c_lat = map_out["last_clicked"]["lat"]
            c_lon = map_out["last_clicked"]["lng"]
            latitude, longitude = c_lat, c_lon
            nearest_loc, dist = find_nearest_locality(latitude, longitude)
            selected_locality = nearest_loc
            st.sidebar.success(f"📍 Nearest: **{nearest_loc}** ({dist:.1f} km)")
            st.sidebar.caption(f"Selected: `{latitude:.4f}, {longitude:.4f}`")
        else:
            selected_locality = "Koramangala"
            latitude, longitude = BENGALURU_LOCALITIES[selected_locality]

    st.sidebar.header("🏡 Property Specifications")
    size = st.sidebar.slider("Bedrooms (BHK)", min_value=1, max_value=5, value=3, step=1)
    total_sqft = st.sidebar.number_input(
        "Super Built-up Area (Sq.Ft)",
        min_value=400.0,
        max_value=5000.0,
        value=1500.0,
        step=50.0,
    )
    bath = st.sidebar.slider("Bathrooms", min_value=1, max_value=5, value=2, step=1)
    balcony = st.sidebar.slider("Balconies", min_value=0, max_value=3, value=1, step=1)

    st.sidebar.header("📝 Description")
    description = st.sidebar.text_area(
        "Property Details",
        value=f"Spacious {size} BHK apartment in {selected_locality} with modern amenities, clubhouse, gym, and covered parking.",
        height=80,
    )

    predict_clicked = st.sidebar.button("🔮 Predict Price", type="primary", use_container_width=True)

    if predict_clicked:
        with st.spinner("Calculating valuation using Random Forest model..."):
            try:
                input_df = pd.DataFrame([{
                    "location": selected_locality,
                    "latitude": latitude,
                    "longitude": longitude,
                    "size": size,
                    "total_sqft": total_sqft,
                    "bath": bath,
                    "balcony": balcony,
                }])

                # Feature engineering
                engineered_df, _ = engineer_features(input_df, kmeans_model=kmeans, is_train=False)

                # Preprocessing
                X_mat = preprocessor.transform(engineered_df)

                # Model Prediction
                log_pred = model.predict(X_mat)
                pred_price = float(inverse_transform_target(log_pred)[0])

                # Store in session state
                st.session_state.predicted_result = {
                    "price": pred_price,
                    "lower": pred_price * 0.90,
                    "upper": pred_price * 1.10,
                    "price_per_sqft": pred_price / total_sqft,
                    "locality": selected_locality,
                    "latitude": latitude,
                    "longitude": longitude,
                    "size": size,
                    "total_sqft": total_sqft,
                    "bath": bath,
                    "balcony": balcony,
                    "description": description,
                }
            except Exception as ex:
                st.error(f"Prediction failed: {ex}")
                logger.error(f"Prediction calculation error: {ex}", exc_info=True)

    # ── Display View ─────────────────────────────────────────────────────────
    res = st.session_state.predicted_result

    if res is not None:
        # Display Results Banner
        st.markdown(f"""
        <div class="price-card">
            <div class="price-title">Estimated Market Value ({res['locality']})</div>
            <div class="price-value">{format_inr(res['price'])}</div>
            <div class="price-subtitle">Estimated Range (±10%): {format_inr(res['lower'])} – {format_inr(res['upper'])}</div>
        </div>
        """, unsafe_allow_html=True)

        # Summary Metrics
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("📍 Location", res["locality"])
        m2.metric("🏠 Layout", f"{res['size']} BHK, {res['bath']} Bath")
        m3.metric("📐 Area", f"{res['total_sqft']:,.0f} sq.ft")
        m4.metric("📊 Rate / Sq.Ft", f"₹{res['price_per_sqft']:,.0f}")

        st.markdown("---")

        # Map & Locality Context
        col_left, col_right = st.columns([3, 2])

        with col_left:
            st.subheader("📍 Property Location & Surroundings")
            prop_map = create_folium_map(res["latitude"], res["longitude"], selected_locality=res["locality"], zoom=13)
            # Use returned_objects=[] so this map never fires unwanted Streamlit reruns
            st_folium(prop_map, width=650, height=400, key="result_display_map", returned_objects=[])

        with col_right:
            st.subheader("📋 Valuation Summary")
            summary_table = pd.DataFrame([
                {"Attribute": "Selected Locality", "Value": str(res["locality"])},
                {"Attribute": "Bedrooms (BHK)", "Value": f"{res['size']} BHK"},
                {"Attribute": "Bathrooms", "Value": f"{res['bath']}"},
                {"Attribute": "Balconies", "Value": f"{res['balcony']}"},
                {"Attribute": "Total Area", "Value": f"{res['total_sqft']:,.0f} sq.ft"},
                {"Attribute": "Valuation Rate", "Value": f"₹{res['price_per_sqft']:,.0f} / sq.ft"},
                {"Attribute": "Algorithm", "Value": "Random Forest Regressor (200 Trees)"},
                {"Attribute": "Training Dataset", "Value": "10,837 Bengaluru Properties (R²=0.9995)"},
            ])
            st.dataframe(summary_table, hide_index=True, use_container_width=True)

    else:
        # Default State: Welcome overview
        st.info("👈 Choose your property location and specs in the sidebar, then click **🔮 Predict Price** to calculate the market valuation.")

        st.subheader("🗺️ Bengaluru Key Localities & IT Corridors")
        overview_map = create_folium_map(BENGALURU_CENTER[0], BENGALURU_CENTER[1], zoom=11)
        st_folium(overview_map, width=950, height=480, key="welcome_overview_map", returned_objects=[])

    st.markdown("---")
    st.caption(f"Bengaluru Real Estate Valuation System | Covering {len(BENGALURU_LOCALITIES)} localities across Bengaluru | Built with Streamlit, scikit-learn, Folium")


if __name__ == "__main__":
    main()
