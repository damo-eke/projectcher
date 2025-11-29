# onboarding_app.py

import streamlit as st
import json
import os

# --- Load real product data from products.json ---
PRODUCTS_PATH = "products.json"

def load_products():
    if os.path.exists(PRODUCTS_PATH):
        with open(PRODUCTS_PATH) as f:
            return json.load(f)
    else:
        return []

products = load_products()

# --- Group products by brand ---
brand_to_products = {}
for item in products:
    brand = item.get("brand", "Unknown Brand")
    brand_to_products.setdefault(brand, []).append(item)

# --- Initialize Streamlit session state ---
if "step" not in st.session_state:
    st.session_state.step = 1
if "selected_brands" not in st.session_state:
    st.session_state.selected_brands = set()

# --- STEP 1: Connect Email ---
if st.session_state.step == 1:
    st.title("📩 Connect your email")
    st.write("We’ll scan your inbox for order confirmations and clothing purchases.")

    st.success("✅ Your email is already connected.")
    if st.button("🔍 Scan Inbox"):
        st.session_state.step += 1

# --- STEP 2: Show detected brands and counts ---
elif st.session_state.step == 2:
    st.title("🧠 Brands detected in your inbox")

    if not brand_to_products:
        st.warning("No products found. Run the email script first to generate `products.json`.")
    else:
        st.write("We found the following brands in your past order confirmations:")
        for brand, items in brand_to_products.items():
            st.checkbox(f"{brand} ({len(items)} items)", key=brand, value=True)

        st.write("You can add more later, don’t worry about catching everything now.")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("← Back"):
                st.session_state.step -= 1
        with col2:
            if st.button("✅ Continue"):
                st.session_state.selected_brands = {
                    brand for brand in brand_to_products if st.session_state.get(brand)
                }
                st.session_state.step += 1

# --- STEP 3: Confirm items to import ---
elif st.session_state.step == 3:
    st.title("🛍️ Items from selected brands")
    st.write("These items are ready to be added to your closet:")

    for brand in st.session_state.selected_brands:
        st.subheader(brand)
        for item in brand_to_products[brand]:
            cols = st.columns([1, 3])
            with cols[0]:
                if "image_url" in item and item["image_url"]:
                    st.image(item["image_url"], use_column_width=True)
                else:
                    st.empty()
            with cols[1]:
                st.markdown(f"**{item.get('product_name', 'Unnamed Item')}**")
                st.caption(f"Color: {item.get('color', '-')}, Size: {item.get('size', '-')}")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("← Back"):
            st.session_state.step -= 1
    with col2:
        if st.button("🧳 Add to Closet"):
            st.session_state.step += 1

# --- STEP 4: Closet Complete ---
elif st.session_state.step == 4:
    st.title("🎉 Closet created!")
    st.write("You’ve added items to your closet. You can now explore your wardrobe anytime.")
    st.success("🧳 Closet setup complete.")
    if st.button("🔁 Start Over"):
        st.session_state.step = 1
        st.session_state.selected_brands = set()
