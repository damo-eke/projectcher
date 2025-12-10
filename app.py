import streamlit as st
import json
import os
import re
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from urllib.parse import urljoin, urlparse

load_dotenv()

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

GOOGLE_CSE_KEY = os.getenv("GOOGLE_CSE_API_KEY")
GOOGLE_CSE_CX = os.getenv("GOOGLE_CSE_CX")


def is_http_image(url: str) -> bool:
    return isinstance(url, str) and url.lower().startswith(("http://", "https://"))


def normalize_image_url(img: str, product_url: str = None) -> str:
    """Clean and resolve relative/quoting issues in image URLs."""
    if not isinstance(img, str):
        return None
    s = img.strip(" '\"")
    if not s:
        return None
    if s.startswith("//"):
        s = "https:" + s
    if s.startswith("/") and product_url:
        parsed = urlparse(product_url)
        if parsed.scheme and parsed.netloc:
            s = urljoin(f"{parsed.scheme}://{parsed.netloc}", s)
    if is_http_image(s):
        return s
    return None


def scrape_og_image(url: str):
    try:
        r = requests.get(url, timeout=15, headers={"User-Agent": "ClosetAI/1.0"})
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        og = soup.find("meta", property="og:image")
        if og and og.get("content"):
            return og["content"]
    except Exception:
        return None
    return None


def google_image_search(query: str, site_domain: str = None):
    if not GOOGLE_CSE_KEY or not GOOGLE_CSE_CX:
        return None
    if site_domain and f"site:{site_domain}" not in query:
        query = f"{query} site:{site_domain}"
    params = {
        "key": GOOGLE_CSE_KEY,
        "cx": GOOGLE_CSE_CX,
        "q": query,
        "searchType": "image",
        "num": 5,
    }
    try:
        r = requests.get("https://www.googleapis.com/customsearch/v1", params=params, timeout=15)
        r.raise_for_status()
        data = r.json()
        items = data.get("items") or []
        if site_domain:
            for it in items:
                link = it.get("link")
                if link and site_domain in link:
                    return link
        if items:
            return items[0].get("link")
    except Exception:
        return None
    return None


def build_image_query(item: dict, site_domain: str = None) -> str:
    parts = [
        item.get("brand") or "",
        item.get("product_name") or "",
        item.get("color") or "",
        item.get("size") or "",
        item.get("style_code") or "",
    ]
    q = " ".join(p for p in parts if p).strip()
    q = re.sub(r"\s+", " ", q)
    if site_domain:
        q = f"{q} site:{site_domain}"
    return q


def pick_best_image(item: dict) -> str:
    # 1) Product page OG image
    product_url = item.get("product_url")
    if product_url and is_http_image(product_url):
        og_img = scrape_og_image(product_url)
        og_img = normalize_image_url(og_img, product_url)
        if og_img:
            return og_img

    # 2) Existing image_url (likely from MCP)
    img = normalize_image_url(item.get("image_url"), product_url)
    if is_http_image(img):
        return img

    # 3) Google image search scoped to brand domain if available
    site_domain = None
    if isinstance(product_url, str):
        match = re.match(r"https?://([^/]+)/", product_url)
        if match:
            site_domain = match.group(1)
    query = build_image_query(item, site_domain=site_domain)
    g_img = google_image_search(query, site_domain=site_domain)
    if g_img:
        return g_img

    return None

# Sidebar filtering
# Normalize brands so sorting does not try to compare None with strings
brands = sorted({str(p.get("brand") or "Unknown") for p in products})
selected_brand = st.sidebar.selectbox("Filter by brand", ["All"] + brands)

filtered = [p for p in products if selected_brand == "All" or p.get("brand") == selected_brand]
st.sidebar.markdown(f"**Showing {len(filtered)} of {len(products)} items**")

# Show results with per-item guard so one bad record doesn't stop the page
errors = 0
grid_cols = st.columns(3)
for idx, item in enumerate(filtered):
    col = grid_cols[idx % 3]
    with col:
        try:
            st.markdown("---")
            img_url = pick_best_image(item)
            if img_url:
                try:
                    st.image(img_url, width=250)
                except Exception as e:
                    st.markdown(f"⚠️ Image unavailable ({e})")
            else:
                st.markdown("❌ No image available")

            st.subheader(item.get("product_name", "Unnamed Product"))
            st.text(f"Brand: {item.get('brand', 'N/A')}")
            st.text(f"Color: {item.get('color', 'N/A')}")
            st.text(f"Size: {item.get('size', 'N/A')}")
            st.text(f"Style Code: {item.get('style_code', 'N/A')}")
            st.text(f"Material: {item.get('material', 'N/A')}")

            if item.get("product_description"):
                st.markdown(f"**Description:** {item['product_description']}")
            elif item.get("editor_note"):
                st.markdown(f"**Editor's Note:** {item['editor_note']}")
            if item.get("features"):
                st.markdown("**Features:**")
                for feature in item["features"]:
                    st.markdown(f"- {feature}")
            if item.get("description"):
                st.caption(item["description"])
            if item.get("product_url"):
                st.markdown(f"[View product page]({item['product_url']})")
        except Exception as e:
            errors += 1

if errors:
    st.warning(f"Skipped {errors} items due to render errors. Check product data for missing/invalid fields.")
