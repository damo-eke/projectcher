import streamlit as st
import json
import os

st.set_page_config(page_title="Closet AI", layout="wide")
st.title("🧾 Closet AI — Product Viewer")

# Load products from file
PRODUCTS_FILE = "products.json"
if not os.path.exists(PRODUCTS_FILE):
    st.warning("No products.json file found. Run the extraction script first.")
    st.stop()

with open(PRODUCTS_FILE, "r") as f:
    products = json.load(f)

if not products:
    st.info("No products found in the file.")
    st.stop()

# Sidebar filtering
brands = sorted(set(p.get("brand", "Unknown") for p in products))
selected_brand = st.sidebar.selectbox("Filter by brand", ["All"] + brands)

filtered = [p for p in products if selected_brand == "All" or p.get("brand") == selected_brand]

# Show results
grid_cols = st.columns(3)
for idx, item in enumerate(filtered):
    col = grid_cols[idx % 3]
    with col:
        st.markdown("---")
        if item.get("image_url"):
            st.image(item["image_url"], width=250)
        else:
            st.markdown("❌ No image available")

        st.subheader(item.get("product_name", "Unnamed Product"))
        st.text(f"Brand: {item.get('brand', 'N/A')}")
        st.text(f"Color: {item.get('color', 'N/A')}")
        st.text(f"Size: {item.get('size', 'N/A')}")
        st.text(f"Style Code: {item.get('style_code', 'N/A')}")
        st.text(f"Material: {item.get('material', 'N/A')}")

        if item.get("editor_note"):
            st.markdown(f"**Editor's Note:** {item['editor_note']}")
        if item.get("features"):
            st.markdown("**Features:**")
            for feature in item["features"]:
                st.markdown(f"- {feature}")
        if item.get("description"):
            st.caption(item["description"])
        if item.get("product_url"):
            st.markdown(f"[View product page]({item['product_url']})")
