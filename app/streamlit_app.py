"""
Streamlit demo app for Bengaluru Real Estate Intelligence System.

Run with: streamlit run app/streamlit_app.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium

from src.data.bengaluru_localities import (
    BENGALURU_LOCALITIES,
    BENGALURU_CENTER,
    find_nearest_locality,
    BENGALURU_REFERENCE_POINTS,
)
from src.utils.currency import format_inr

# Page config
st.set_page_config(
    page_title="Bengaluru Real Estate Intelligence",
    page_icon="🏠",
    layout="wide",
)


def create_bengaluru_map(center_lat, center_lon, selected_locality=None):
    """Create an interactive Folium map of Bengaluru with locality markers."""
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=12,
        tiles="OpenStreetMap",
    )

    # Add selected location marker
    if selected_locality:
        folium.Marker(
            [center_lat, center_lon],
            popup=f"<b>{selected_locality}</b>",
            tooltip=selected_locality,
            icon=folium.Icon(color="red", icon="home", prefix="fa"),
        ).add_to(m)

    # Add IT Hub markers
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
    **Hybrid ML + NLP + Retrieval + Explainability**

    Predicts Bengaluru property prices using structured features and
    property descriptions with ML + DistilBERT embeddings.
    """)

    st.sidebar.header("Property Location")

    # Location Selection Mode
    location_mode = st.sidebar.radio(
        "Choose location method:",
        ["🔍 Select from Locality List", "🗺️ Click on Map"],
        index=0,
    )

    selected_locality = None
    latitude, longitude = BENGALURU_CENTER

    if location_mode == "🔍 Select from Locality List":
        # Dropdown selection
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
        # Map-based selection
        st.sidebar.info("👇 Click anywhere on the map to select location")

        # Create map
        map_obj = create_bengaluru_map(BENGALURU_CENTER[0], BENGALURU_CENTER[1])

        # Display map in sidebar (smaller)
        map_data = st_folium(
            map_obj,
            width=350,
            height=400,
            key="location_map",
        )

        # Get clicked location
        if map_data and map_data.get("last_clicked"):
            clicked_lat = map_data["last_clicked"]["lat"]
            clicked_lon = map_data["last_clicked"]["lng"]
            latitude, longitude = clicked_lat, clicked_lon

            # Find nearest locality
            nearest_loc, distance = find_nearest_locality(latitude, longitude)
            selected_locality = nearest_loc

            st.sidebar.success(f"📍 **Nearest: {nearest_loc}**")
            st.sidebar.caption(f"({distance:.2f} km from {nearest_loc})")
            st.sidebar.caption(f"Coordinates: {latitude:.4f}, {longitude:.4f}")
        else:
            selected_locality = "Koramangala"
            latitude, longitude = BENGALURU_LOCALITIES[selected_locality]
            st.sidebar.info(f"Default: {selected_locality}")

    # Property Details
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

    st.sidebar.header("📝 Property Description")
    property_description = st.sidebar.text_area(
        "Describe the property",
        value=f"Spacious {size} BHK apartment in {selected_locality} with modern amenities, "
              f"clubhouse, gym, and covered parking. Close to major IT parks and excellent connectivity.",
        height=120,
    )

    # Predict button
    if st.sidebar.button("🔮 Predict Price", type="primary", use_container_width=True):
        property_data = {
            "location": selected_locality,
            "latitude": latitude,
            "longitude": longitude,
            "size": size,
            "total_sqft": total_sqft,
            "bath": bath,
            "balcony": balcony,
            "property_description": property_description,
        }

        with st.spinner("🔄 Calculating property value..."):
            # Placeholder prediction (will be replaced with ML model)
            base_price_per_sqft = 5500

            # Adjust by BHK size
            bhk_multiplier = 1 + (size - 2) * 0.1

            # Calculate estimated price
            estimated_price = total_sqft * base_price_per_sqft * bhk_multiplier

            # Price interval (±15%)
            lower_price = estimated_price * 0.85
            upper_price = estimated_price * 1.15

        # Display Prediction Results
        st.success("✅ **Prediction Complete!**")

        # Main price display
        st.markdown("---")
        st.markdown("### 💰 Predicted Property Price")

        price_col1, price_col2, price_col3 = st.columns([2, 2, 1])

        with price_col1:
            st.markdown(f"## {format_inr(estimated_price)}")
            st.caption("Estimated Market Value")

        with price_col2:
            st.metric(
                "Price Range (±15%)",
                f"{format_inr(lower_price)} - {format_inr(upper_price)}",
            )

        with price_col3:
            st.metric(
                "Per Sqft",
                f"₹{base_price_per_sqft * bhk_multiplier:,.0f}",
            )

        st.info("ℹ️ **Note:** This is a basic estimate. Full ML model with DistilBERT embeddings + SHAP explanations coming after model retraining.")

        # Display input summary
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

        # Show map in main area
        st.markdown("---")
        st.subheader("📍 Property Location on Map")
        main_map = create_bengaluru_map(latitude, longitude, selected_locality)
        st_folium(main_map, width=None, height=450, key="main_map")

        # Similar properties placeholder
        st.markdown("---")
        st.subheader("🔍 Similar Properties (Coming Soon)")
        st.info("After model retraining, this will show comparable properties in the same area with actual prices and similarity scores.")

    else:
        st.info("👈 Configure property details in the sidebar and click **Predict Price**")

        # Show Bengaluru map with IT hubs
        st.subheader("🗺️ Bengaluru IT Hubs & Major Localities")
        overview_map = create_bengaluru_map(BENGALURU_CENTER[0], BENGALURU_CENTER[1])
        st_folium(overview_map, width=900, height=500, key="overview_map")

    # Footer
    st.markdown("---")
    st.markdown(f"""
    **About**: Bengaluru Real Estate Intelligence System predicts property prices using:
    - **Structured Features**: BHK, sqft, bathrooms, balconies, location
    - **Distance Engineering**: Proximity to CBD, IT hubs, Airport
    - **NLP Embeddings**: DistilBERT on property descriptions
    - **Explainability**: SHAP values for transparent predictions

    **Coverage**: {len(BENGALURU_LOCALITIES)} localities across North, South, East, West Bengaluru

    Built with: Python, scikit-learn, PyTorch, Transformers, SHAP, Streamlit, Folium
    """)


if __name__ == "__main__":
    main()
