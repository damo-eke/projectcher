#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import json
import math
import base64
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from openai import OpenAI

# -------------- Config / Setup --------------

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_CSE_KEY = os.getenv("GOOGLE_CSE_API_KEY")
GOOGLE_CSE_CX = os.getenv("GOOGLE_CSE_CX")

if not OPENAI_API_KEY:
    raise RuntimeError("Missing OPENAI_API_KEY")
if not GOOGLE_CSE_KEY or not GOOGLE_CSE_CX:
    print("⚠️  Missing GOOGLE_CSE_API_KEY or GOOGLE_CSE_CX; Google enrichment may fail.")

client = OpenAI(api_key=OPENAI_API_KEY)
SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']

SUPPORTED_SENDERS = {
    "no-reply@kithnyc.com": "kith",
    "orders@aimeleondore.com": "ald",
    "service@em.shopbop.com": "shopbop"
}

STORE_DOMAINS = {
    "kith": "kith.com",
    "ald": "aimeleondore.com",
    "shopbop": "shopbop.com"
}

DEFAULT_HEADERS = {
    "Content-Type": "application/json",
    "Accept": "application/json",
    "User-Agent": "ClosetAI/1.0"
}

KITH_COLOR_CODES = {
    "101": "Black",
    "102": "White",
    "103": "Red",
    "104": "Veil",
    "105": "Sandrift",
    "106": "Navy",
    "107": "Slate",
    "108": "Woodrose",
}
# Reverse map: name -> code
KITH_COLOR_TO_CODE = {v: k for k, v in KITH_COLOR_CODES.items()}

# -------------- Small utils --------------

def _simplify_query(q: str) -> str:
    """Remove slashes/extra fluff; cap to ~8 words; collapse spaces."""
    q = q.replace("/", " ")
    q = re.sub(r"\s+", " ", q).strip()
    parts = q.split()
    if len(parts) > 8:
        q = " ".join(parts[:8])
    return q

def _coerce_json(val):
    """Turn a stringified JSON blob into dict/list when possible."""
    if isinstance(val, (dict, list)):
        return val
    if isinstance(val, str):
        try:
            return json.loads(val)
        except Exception:
            try:
                return json.loads(bytes(val, "utf-8").decode("unicode_escape"))
            except Exception:
                return None
    return None

def _norm(s):
    return (s or "").strip().lower()

def _strip_brand_and_color_from_title(name: str, brand: str, color: str) -> str:
    """
    From 'KITH LONG SLEEVE LAX TEE - VEIL' -> 'Long Sleeve Lax Tee'
    """
    t = name or ""
    if brand:
        t = re.sub(re.escape(brand), "", t, flags=re.IGNORECASE)
    if color:
        t = re.sub(rf"\s*[-–]\s*{re.escape(color)}\s*$", "", t, flags=re.IGNORECASE)
    t = re.sub(r"\s+", " ", t).strip()
    t = re.sub(r"^\W+|\bkith\b", "", t, flags=re.IGNORECASE).strip()
    return t

def parse_money(val):
    """Return (amount_float, currency) from strings like '$75.00', '75', or dicts like {'amount':'75','currencyCode':'USD'}."""
    if isinstance(val, dict):
        amt = val.get("amount") or val.get("price") or val.get("value")
        cur = val.get("currency") or val.get("currencyCode") or val.get("currency_code")
        try:
            return (float(amt), cur)
        except Exception:
            return (None, cur)
    if isinstance(val, (int, float)):
        return (float(val), None)
    if isinstance(val, str):
        cur = None
        m = re.search(r"([A-Za-z]{3})", val)
        if m:
            cur = m.group(1).upper()
        s = re.sub(r"[^\d\.\-]", "", val)
        try:
            return (float(s), cur)
        except Exception:
            return (None, cur)
    return (None, None)

def approx_equal(a, b, tol=1.5):
    if a is None or b is None:
        return False
    return math.fabs(a - b) <= tol

# -------------- Shopify MCP: tools/list + search --------------

def mcp_list_tools(store_domain: str, timeout: int = 15):
    endpoint = f"https://{store_domain}/api/mcp"
    payload = {"jsonrpc": "2.0", "id": "list", "method": "tools/list"}
    try:
        r = requests.post(endpoint, headers=DEFAULT_HEADERS, json=payload, timeout=timeout)
        if r.status_code != 200:
            return (False, None, {"status": r.status_code, "text": r.text[:400]})
        data = r.json()
        if "result" in data:
            return (True, data.get("result"), data)
        return (False, None, data)
    except Exception as e:
        return (False, None, {"error": str(e)})

def mcp_search(search_term: str,
               store_domain: str,
               context: str = "Order email enrichment: return product details including variants"):
    """
    Calls Storefront MCP search_shop_catalog. Returns (normalized_items, raw_items).
    Handles result.content[].text stringified JSON and simplified retry.
    """
    endpoint = f"https://{store_domain}/api/mcp"

    def _normalize(p: dict):
        title = p.get("title") or p.get("name") or ""
        handle = p.get("handle") or p.get("product_handle") or ""
        url = p.get("url") or p.get("product_url") or (f"https://{store_domain}/products/{handle}" if handle else "")
        image = p.get("image") or p.get("image_url") or ""
        description = p.get("description") or p.get("body_html") or p.get("excerpt") or ""
        price = p.get("price") or p.get("amount")
        currency = p.get("currency") or p.get("price_currency")
        variant_id = p.get("variant_id") or p.get("merchandise_id")
        return {
            "title": title, "handle": handle, "url": url, "image": image,
            "description": description, "price": price, "currency": currency,
            "variant_id": variant_id
        }

    def _call(q):
        payload = {
            "jsonrpc": "2.0",
            "method": "tools/call",
            "id": 1,
            "params": {
                "name": "search_shop_catalog",
                "arguments": {"query": q, "context": context}
            }
        }
        r = requests.post(endpoint, headers=DEFAULT_HEADERS, json=payload, timeout=20)
        print(f"🛠️ MCP Debug: Searching '{q}' @ {store_domain} (ctx: {context})")
        print(f"🛠️ MCP Response [{r.status_code}]: {r.text[:300]}")
        r.raise_for_status()
        data = r.json()
        if "error" in data:
            raise RuntimeError(data["error"])

        res = data.get("result", {})
        raw_items = []

        if isinstance(res, dict) and isinstance(res.get("items"), list):
            raw_items = res["items"]
        elif isinstance(res, dict) and isinstance(res.get("content"), list):
            for entry in res["content"]:
                if entry.get("type") == "output":
                    out = entry.get("output")
                    if isinstance(out, list):
                        raw_items.extend(out)
                    elif isinstance(out, dict):
                        raw_items.append(out)
                elif entry.get("type") == "text":
                    parsed = _coerce_json(entry.get("text", ""))
                    if isinstance(parsed, dict):
                        if isinstance(parsed.get("products"), list):
                            raw_items.extend(parsed["products"])
                        elif isinstance(parsed.get("items"), list):
                            raw_items.extend(parsed["items"])
                    elif isinstance(parsed, list):
                        raw_items.extend(parsed)
        elif isinstance(res, list):
            raw_items = res

        norm = [_normalize(p) for p in raw_items]
        return norm, raw_items

    # full query then simplified
    norm, raw = _call(search_term)
    if not norm:
        simple = _simplify_query(search_term)
        if simple and simple.lower() != search_term.lower():
            print(f"🔁 Retrying MCP with simplified query: {simple}")
            norm, raw = _call(simple)

    if norm:
        print("✅ MCP first match:", json.dumps(norm[0], indent=2)[:800])

    return norm, raw

# -------------- Shopify search/suggest → handles --------------

def shopify_search_suggest(store_domain: str, query: str, limit: int = 10):
    """
    Uses Shopify's public search suggest to fetch likely product handles.
    We'll then re-query MCP by handle for structured data.
    """
    try:
        endpoint = f"https://{store_domain}/search/suggest.json"
        params = {
            "q": query,
            "resources[type]": "product",
            "resources[limit]": str(limit),
        }
        r = requests.get(endpoint, params=params, timeout=12,
                         headers={"User-Agent": DEFAULT_HEADERS["User-Agent"]})
        print(f"🧭 suggest.json → {r.url}")
        r.raise_for_status()
        data = r.json()
        handles = []

        res = (data.get("resources") or {}).get("results") or {}
        products = res.get("products") or res.get("products_unavailable") or []
        for p in products:
            h = p.get("handle")
            if not h:
                url = p.get("url") or p.get("path") or ""
                if "/products/" in url:
                    h = url.rsplit("/products/", 1)[-1].strip("/ ")
            if h:
                handles.append(h)

        seen = set(); uniq = []
        for h in handles:
            if h not in seen:
                uniq.append(h); seen.add(h)
        return uniq

    except Exception as e:
        print(f"❌ search/suggest failed: {e}")
        return []

# -------------- Variant selection (color/size/price aware) --------------

def _extract_variants(product_raw: dict):
    for key in ["variants", "merchandises", "merchandise", "skus", "items"]:
        v = product_raw.get(key)
        if isinstance(v, list) and v:
            return v
    for key in ["product", "node"]:
        if isinstance(product_raw.get(key), dict):
            return _extract_variants(product_raw[key])
    return []

def _variant_options(v):
    opts = {}
    so = v.get("selectedOptions") or v.get("selected_options") or v.get("options")
    if isinstance(so, list):
        for o in so:
            name = _norm(o.get("name"))
            val = (o.get("value") or "").strip()
            if name:
                opts[name] = val
    vt = v.get("title") or ""
    if "/" in vt:
        parts = [p.strip() for p in vt.split("/")]
        for p in parts:
            if p.upper() in ["XS","S","M","L","XL","XXL","XXXL"]:
                opts.setdefault("size", p)
            else:
                if "color" not in opts:
                    opts.setdefault("color", p)
    return opts

def pick_best_variant(product_raw: dict, desired_color: str, desired_size: str, desired_price):
    d_color = _norm(desired_color)
    d_size = _norm(desired_size)
    d_price, _ = parse_money(desired_price)
    best, best_score = None, -1
    variants = _extract_variants(product_raw)
    for v in variants:
        opts = _variant_options(v)
        v_color = _norm(opts.get("color") or opts.get("colour") or "")
        v_size  = _norm(opts.get("size") or "")
        v_price_raw = v.get("price") or v.get("priceV2") or v.get("presentmentPrices")
        if isinstance(v_price_raw, list) and v_price_raw:
            v_price_raw = v_price_raw[0]
        v_price, _ = parse_money(v_price_raw)
        score = 0
        if d_color and (d_color == v_color or d_color in _norm(v.get("title","")) or d_color in _norm(v.get("sku","") or "")):
            score += 2
        if d_size and d_size == v_size:
            score += 2
        if d_price is not None and v_price is not None and approx_equal(d_price, v_price, tol=2.0):
            score += 1
        if v.get("available") is True:
            score += 0.1
        if score > best_score:
            best, best_score = v, score
    return best, best_score

# -------------- Google + HTML helpers --------------

def google_image_search(query: str):
    endpoint = "https://www.googleapis.com/customsearch/v1"
    params = {"q": query, "key": GOOGLE_CSE_KEY, "cx": GOOGLE_CSE_CX, "searchType": "image", "num": 1}
    try:
        r = requests.get(endpoint, params=params, timeout=15, headers={"User-Agent": DEFAULT_HEADERS["User-Agent"]})
        print(f"🔍 Google Image URL: {r.url}")
        print(f"🔍 Google Image Response: {r.text[:200]}")
        r.raise_for_status()
        data = r.json()
        if data.get("items"):
            return data["items"][0]["link"]
    except Exception as e:
        print(f"❌ Google image search failed: {e}")
    return None

def google_web_search(query: str):
    endpoint = "https://www.googleapis.com/customsearch/v1"
    params = {"q": query, "key": GOOGLE_CSE_KEY, "cx": GOOGLE_CSE_CX, "num": 1}
    try:
        r = requests.get(endpoint, params=params, timeout=15, headers={"User-Agent": DEFAULT_HEADERS["User-Agent"]})
        print(f"🔎 Google Web URL: {r.url}")
        print(f"🔎 Google Web Response: {r.text[:200]}")
        r.raise_for_status()
        data = r.json()
        if data.get("items"):
            return data["items"][0]["link"]
    except Exception as e:
        print(f"❌ Google web search failed: {e}")
    return None

def scrape_og_image(url: str):
    try:
        print(f"🕵️ Scraping OG image from: {url}")
        r = requests.get(url, timeout=15, headers={"User-Agent": DEFAULT_HEADERS["User-Agent"]})
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        og = soup.find("meta", property="og:image")
        if og and og.get("content"):
            return og["content"]
    except Exception as e:
        print(f"❌ Failed to scrape OG image from {url}: {e}")
    return None

# -------------- Brand-specific helpers (Kith handle/SKU) --------------

KITH_CODE_RE = re.compile(r'\bkh[a-z]\d{6}\b', re.I)
HANDLE_FROM_URL_RE = re.compile(r'/products/([a-z0-9\-]+)', re.I)

def extract_kith_style_code(text: str):
    if not text:
        return None
    m = KITH_CODE_RE.search(text)
    return m.group(0).lower() if m else None

def extract_handle_from_url(url: str):
    m = HANDLE_FROM_URL_RE.search(url or "")
    return m.group(1).lower() if m else None

# -------------- Email parsing + OpenAI --------------

def get_email_body(message: dict):
    payload = message.get("payload", {})
    parts = payload.get("parts", []) or []
    body_data = payload.get("body", {}).get("data")
    if body_data:
        decoded = base64.urlsafe_b64decode(body_data.encode('utf-8')).decode('utf-8', errors="ignore")
        return BeautifulSoup(decoded, 'html.parser').get_text(separator="\n")
    for part in parts:
        mime = part.get("mimeType")
        data = part.get("body", {}).get("data")
        if not data:
            continue
        decoded = base64.urlsafe_b64decode(data.encode('utf-8')).decode('utf-8', errors="ignore")
        if mime == "text/html":
            return BeautifulSoup(decoded, 'html.parser').get_text(separator="\n")
        if mime == "text/plain":
            return decoded
    return ""

def extract_product_metadata(email_text: str):
    prompt = f"""
Extract structured clothing order info from the following email.
Return ONLY valid JSON (no backticks, no prose), shaped as a list of objects:
[
  {{
    "product_name": "...",
    "brand": "...",
    "color": "...",
    "size": "..."
  }}
]

Email:
{email_text}
""".strip()

    try:
        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
        content = resp.choices[0].message.content.strip()
        if content.startswith("```"):
            content = re.sub(r"^```[a-zA-Z]*\n", "", content)
            content = re.sub(r"\n```$", "", content).strip()
        if not content.startswith("["):
            print("⚠️ OpenAI returned unexpected response (not JSON array).")
            print(content[:500])
            return []
        data = json.loads(content)
        if isinstance(data, dict):
            data = [data]
        return data
    except Exception as e:
        print(f"❌ Failed to extract metadata from email: {e}")
        return []

def build_search_query_from_item(item: dict):
    brand = (item.get("brand") or "").strip()
    name = (item.get("product_name") or "").strip()
    color = (item.get("color") or "").strip()

    if brand and brand.lower() in name.lower():
        name = re.sub(re.escape(brand), "", name, flags=re.IGNORECASE).strip()

    name = name.replace(" -", "").strip()
    color = color.replace("-", " ").strip()

    query = " ".join([brand, name, color]).strip()
    return re.sub(r"\s+", " ", query)

# -------------- Main Flow --------------

def main():
    # Gmail auth
    creds = None
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    if not creds or not creds.valid:
        if creds and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        with open('token.json', 'w') as token:
            token.write(creds.to_json())

    service = build('gmail', 'v1', credentials=creds)

    print("📨 Searching for order confirmation emails...")
    results = service.users().messages().list(
        userId='me',
        maxResults=50,
        q='from:(no-reply@kithnyc.com OR orders@aimeleondore.com OR service@em.shopbop.com) subject:(order OR shipped OR delivered OR receipt)'
    ).execute()

    messages = results.get('messages', [])
    print(f"📬 Found {len(messages)} matching emails.")
    all_products = []
    seen = set()

    for msg in messages:
        try:
            msg_data = service.users().messages().get(userId='me', id=msg['id'], format='full').execute()
            headers = msg_data['payload'].get('headers', [])
            from_header = next((h['value'] for h in headers if h['name'].lower() == 'from'), '')
            subject_header = next((h['value'] for h in headers if h['name'].lower() == 'subject'), '')
            sender_email_match = re.search(r"<(.+?)>", from_header)
            sender = sender_email_match.group(1).lower() if sender_email_match else from_header.strip().lower()

            if sender not in SUPPORTED_SENDERS:
                print(f"⏭️ Skipping unsupported sender: {sender}")
                continue

            brand_key = SUPPORTED_SENDERS[sender]
            store_domain = STORE_DOMAINS.get(brand_key, "")

            print(f"📌 Processing email from {sender}: {subject_header}")
            body_text = get_email_body(msg_data)

            extracted = extract_product_metadata(body_text)
            print("📦 Extracted metadata:", json.dumps(extracted, indent=2))

            for item in extracted:
                if sender == "no-reply@kithnyc.com" and "receipt" in subject_header.lower():
                    item["brand"] = "Kith"

                color = (item.get("color") or "").strip()
                if color.isdigit() and color in KITH_COLOR_CODES:
                    item["color"] = KITH_COLOR_CODES[color]

                unique_id = f"{(item.get('brand','')).lower()}|{(item.get('product_name','')).lower()}|{(item.get('color','')).lower()}|{(item.get('size','')).lower()}"
                if unique_id in seen:
                    print(f"🛑 Skipping duplicate item: {unique_id}")
                    continue
                seen.add(unique_id)

                # ---------------- Build queries ----------------
                search_query = build_search_query_from_item(item)
                search_query = _simplify_query(search_query)

                base_title_query = _strip_brand_and_color_from_title(
                    item.get("product_name",""),
                    item.get("brand",""),
                    item.get("color","")
                )

                enriched = dict(item)
                enriched["image_url"] = None

                # Quick image try
                enriched["image_url"] = enriched["image_url"] or google_image_search(
                    " ".join([item.get("brand",""), item.get("product_name",""), item.get("color","")]).strip()
                )

                # ---------- MCP FIRST ----------
                mcp_norm, mcp_raw = [], []

                if store_domain:
                    # Kith: try style-code/handle first
                    if brand_key == "kith":
                        style_code = extract_kith_style_code(body_text) or extract_kith_style_code(search_query)
                        color_name = (item.get("color") or "").strip()
                        color_code = KITH_COLOR_TO_CODE.get(color_name) if color_name else None

                        if style_code and color_code:
                            n, r = mcp_search(f"{style_code}-{color_code}", store_domain,
                                              context="Enrich by handle")
                            mcp_norm, mcp_raw = n or [], r or []
                        if not mcp_norm and style_code:
                            n, r = mcp_search(style_code, store_domain, context="Enrich by style code")
                            mcp_norm, mcp_raw = n or [], r or []

                    # If still nothing, try base title query (brand/color stripped)
                    if not mcp_norm:
                        n, r = mcp_search(base_title_query or search_query, store_domain,
                                          context="Title-based enrichment including variants")
                        mcp_norm, mcp_raw = n or [], r or []

                    # NEW: If still nothing, use Shopify search/suggest to get handles → MCP by handle
                    if not mcp_norm:
                        suggest_q = base_title_query or search_query
                        handles = shopify_search_suggest(store_domain, suggest_q, limit=8)
                        print(f"🧭 suggest handles for '{suggest_q}': {handles}")
                        for h in handles:
                            n, r = mcp_search(h, store_domain, context="Enrich by handle from search/suggest")
                            if n:
                                mcp_norm, mcp_raw = n, r
                                break

                picked = False
                if mcp_raw:
                    # Pick best variant by color/size/price
                    variant, score = (None, -1)
                    best_product = None
                    for product in mcp_raw:
                        v, s = pick_best_variant(
                            product_raw=product,
                            desired_color=item.get("color",""),
                            desired_size=item.get("size",""),
                            desired_price=item.get("price") or item.get("Price") or None
                        )
                        if s > (score or -1):
                            variant, score = v, s
                            best_product = product
                    if variant:
                        handle = best_product.get("handle") or extract_handle_from_url(
                            (best_product.get("url") or best_product.get("product_url") or "")
                        ) or ""
                        base_url = f"https://{store_domain}/products/{handle}" if handle else \
                                   (best_product.get("url") or best_product.get("product_url"))
                        enriched["product_url"] = base_url
                        var_id = variant.get("id") or variant.get("variant_id") or variant.get("merchandiseId") or variant.get("merchandise_id")
                        if base_url and var_id:
                            enriched["product_url"] = f"{base_url}?variant={var_id}"

                        vimg = variant.get("image")
                        if isinstance(vimg, dict):
                            vimg = vimg.get("url") or vimg.get("src")
                        enriched["image_url"] = enriched.get("image_url") or vimg or best_product.get("image")

                        enriched["editor_note"] = best_product.get("description") or best_product.get("body_html")
                        vprice = variant.get("price") or variant.get("priceV2")
                        p_amt, p_cur = parse_money(vprice)
                        if p_amt is None and "price" in best_product:
                            p_amt, p_cur = parse_money(best_product.get("price"))
                        enriched["price"] = p_amt
                        enriched["currency"] = p_cur
                        enriched["variant_id"] = var_id
                        picked = True

                # ---------- If MCP didn't give us a variant, fall back ----------
                if not picked:
                    if mcp_norm:
                        best = mcp_norm[0]
                        enriched["product_url"] = best.get("url")
                        enriched["image_url"] = enriched.get("image_url") or best.get("image")
                        enriched["editor_note"] = best.get("description")
                        enriched["price"] = best.get("price")
                        enriched["currency"] = best.get("currency")
                        enriched["variant_id"] = best.get("variant_id")
                    else:
                        print("⚠️ MCP returned 0 results; using Google + scraping fallback.")
                        if brand_key == "kith":
                            page = google_web_search(f"{base_title_query or search_query} site:kith.com/products")
                            if page and "kith.com/products" in page:
                                enriched["product_url"] = page
                                scraped_img = scrape_og_image(page)
                                if scraped_img:
                                    enriched["image_url"] = scraped_img
                                # Try enriching by handle from URL
                                handle = extract_handle_from_url(enriched.get("product_url"))
                                if handle:
                                    n, r = mcp_search(handle, store_domain, context="Enrich by handle from URL")
                                    if n:
                                        best = n[0]
                                        enriched["product_url"] = best.get("url") or enriched.get("product_url")
                                        enriched["image_url"] = enriched.get("image_url") or best.get("image")
                                        enriched["editor_note"] = best.get("description") or enriched.get("editor_note")
                                        enriched["price"] = best.get("price") or enriched.get("price")
                                        enriched["currency"] = best.get("currency") or enriched.get("currency")
                                        enriched["variant_id"] = best.get("variant_id")

                        elif brand_key == "ald":
                            page = google_web_search(f"{base_title_query or search_query} site:aimeleondore.com/products")
                            if page and "aimeleondore.com/products" in page:
                                enriched["product_url"] = page
                                scraped_img = scrape_og_image(page)
                                if scraped_img:
                                    enriched["image_url"] = scraped_img

                        elif brand_key == "shopbop":
                            page = google_web_search(f"{base_title_query or search_query} site:shopbop.com")
                            if page and "shopbop.com" in page:
                                enriched["product_url"] = page
                                scraped_img = scrape_og_image(page)
                                if scraped_img:
                                    enriched["image_url"] = scraped_img

                # Last-resort image
                if not enriched.get("image_url"):
                    enriched["image_url"] = google_image_search(
                        " ".join([item.get("brand",""), base_title_query or item.get("product_name",""), item.get("color","")]).strip()
                    )

                all_products.append(enriched)

        except Exception as e:
            print(f"⚠️ Error processing message: {e}")

    print(f"🧾 Total products extracted: {len(all_products)}")
    if all_products:
        with open("products.json", "w") as f:
            json.dump(all_products, f, indent=2)
        print("✅ Saved product metadata to products.json")
    else:
        print("⚠️ No products to save.")

if __name__ == "__main__":
    main()
